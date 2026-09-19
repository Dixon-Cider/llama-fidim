//! Saved conversations and per-profile chat presets.
//!
//! One JSON file per conversation under `<config-dir>/chats/<id>.json`, and
//! presets (system prompt, sampler overrides, thinking) keyed by profile id
//! in `<config-dir>/chat/presets.json`, kept apart from the profiles so a
//! profile stays launch configuration. Every write goes to a temporary file
//! in the same folder and is renamed over the old one, so a crash mid-write
//! never leaves half a conversation behind.
//!
//! The conversation schema belongs to the GUI; this side checks only what it
//! needs to file it (a safe id) and to list it (title, times, target).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use serde_json::Value;

use crate::config::Config;
use crate::{Error, Result};

/// Refuse to write a conversation bigger than this: a runaway client, not a chat.
const MAX_CONV_BYTES: usize = 64 * 1024 * 1024;
const MAX_PRESETS_BYTES: usize = 4 * 1024 * 1024;

pub fn chats_dir() -> PathBuf {
    Config::config_dir().join("chats")
}

pub fn presets_path() -> PathBuf {
    Config::config_dir().join("chat").join("presets.json")
}

/// Conversation ids become file names: letters, digits, `-` and `_` only.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn conv_path(dir: &Path, id: &str) -> Result<PathBuf> {
    if !valid_id(id) {
        return Err(Error::Config(format!("invalid conversation id {id:?}")));
    }
    Ok(dir.join(format!("{id}.json")))
}

/// Write `bytes` to `path` through a temporary sibling and a rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().ok_or_else(|| Error::Config(format!("{} has no folder", path.display())))?;
    std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.{}-{}.tmp", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    if let Err(e) = std::fs::write(&tmp, bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::io(&tmp, e));
    }
    // std's rename replaces an existing file on Windows (MOVEFILE_REPLACE_EXISTING).
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::io(path, e)
    })
}

/// One row of the conversation list.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConvSummary {
    pub id: String,
    pub title: String,
    pub created_unix: u64,
    pub updated_unix: u64,
    pub target: Value,
    pub n_messages: usize,
}

fn summary_of(v: &Value) -> Option<ConvSummary> {
    let id = v.get("id")?.as_str()?.to_string();
    valid_id(&id).then(|| ConvSummary {
        title: v.get("title").and_then(Value::as_str).unwrap_or("").to_string(),
        created_unix: v.get("created_unix").and_then(Value::as_u64).unwrap_or(0),
        updated_unix: v.get("updated_unix").and_then(Value::as_u64).unwrap_or(0),
        target: v.get("target").cloned().unwrap_or(Value::Null),
        n_messages: v.get("messages").and_then(Value::as_array).map_or(0, Vec::len),
        id,
    })
}

/// Every `*.json` file in the folder that holds a conversation (an object
/// with an id and messages), whatever the file is called.
fn conv_files(dir: &Path) -> Vec<(PathBuf, Value)> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")))
        .filter_map(|p| {
            let t = std::fs::read_to_string(&p).ok()?;
            let v = serde_json::from_str::<Value>(t.trim_start_matches('\u{feff}')).ok()?;
            let is_conv = v.get("id").is_some_and(Value::is_string) && v.get("messages").is_some_and(Value::is_array);
            is_conv.then_some((p, v))
        })
        .collect()
}

/// Every readable conversation, most recently updated first. Only a file
/// named after the id inside it is listed: `load` and `delete` find a
/// conversation by that name, so a copy made in Explorer (`<id> - Copy.json`)
/// or a renamed file would be a row that cannot be opened or deleted. A
/// file that does not parse is skipped, never deleted: it may be a newer
/// schema.
pub fn list(dir: &Path) -> Vec<ConvSummary> {
    let mut out: Vec<ConvSummary> = conv_files(dir)
        .into_iter()
        // Windows file names ignore case, and so does `load`.
        .filter_map(|(p, v)| summary_of(&v).filter(|s| p.file_stem().is_some_and(|n| n.eq_ignore_ascii_case(&s.id))))
        .collect();
    out.sort_by(|a, b| b.updated_unix.cmp(&a.updated_unix).then_with(|| a.id.cmp(&b.id)));
    out
}

pub fn load(dir: &Path, id: &str) -> Result<Value> {
    let p = conv_path(dir, id)?;
    let text = std::fs::read_to_string(&p).map_err(|e| Error::io(&p, e))?;
    Ok(serde_json::from_str(text.trim_start_matches('\u{feff}'))?)
}

/// Save a conversation object under its own `id`.
pub fn save(dir: &Path, conv: &Value) -> Result<()> {
    if !conv.is_object() {
        return Err(Error::Config("a conversation must be a JSON object".into()));
    }
    let id = conv.get("id").and_then(Value::as_str).unwrap_or("");
    let p = conv_path(dir, id)?;
    let text = serde_json::to_string_pretty(conv)?;
    if text.len() > MAX_CONV_BYTES {
        return Err(Error::Config(format!("conversation {id} is larger than {} MB", MAX_CONV_BYTES >> 20)));
    }
    write_atomic(&p, text.as_bytes())
}

/// Remove one conversation; false when it was not there.
pub fn delete(dir: &Path, id: &str) -> Result<bool> {
    let p = conv_path(dir, id)?;
    match std::fs::remove_file(&p) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(Error::io(&p, e)),
    }
}

/// Remove every saved conversation: each file that holds one, by its path,
/// so copies and renamed files go too, and the temporary files a crash
/// mid-save left behind (never one this process is writing now). Settings
/// calls it to take every prompt off the disk.
pub fn delete_all(dir: &Path) -> Result<usize> {
    let mut n = 0;
    for (p, _) in conv_files(dir) {
        match std::fs::remove_file(&p) {
            Ok(()) => n += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::io(&p, e)),
        }
    }
    let ours = format!(".{}-", std::process::id());
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        // `write_atomic`'s `.<name>.<pid>-<n>.tmp`.
        if name.starts_with('.') && name.ends_with(".tmp") && !name.contains(&ours) && e.path().is_file() {
            let _ = std::fs::remove_file(e.path());
        }
    }
    Ok(n)
}

/// The presets object (`{}` when there is none yet or it does not parse).
pub fn load_presets(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(t.trim_start_matches('\u{feff}')).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(Default::default()))
}

pub fn save_presets(path: &Path, presets: &Value) -> Result<()> {
    if !presets.is_object() {
        return Err(Error::Config("chat presets must be a JSON object keyed by profile id".into()));
    }
    let text = serde_json::to_string_pretty(presets)?;
    if text.len() > MAX_PRESETS_BYTES {
        return Err(Error::Config("chat presets are too large".into()));
    }
    write_atomic(path, text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fidim-chat-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn conv(id: &str, updated: u64) -> Value {
        json!({
            "schema": 1, "id": id, "title": format!("chat {id}"), "created_unix": 1, "updated_unix": updated,
            "target": { "run": "worker-pool", "model": null, "engine": "llama-server" },
            "system_prompt": "", "params": {}, "thinking": null,
            "messages": [{ "id": "m1", "role": "user", "content": "hi" }],
        })
    }

    #[test]
    fn save_list_load_delete_round_trip() {
        let dir = temp("rt");
        assert!(list(&dir).is_empty(), "a missing folder lists nothing");
        save(&dir, &conv("a1", 10)).unwrap();
        save(&dir, &conv("b2", 30)).unwrap();
        save(&dir, &conv("c3", 20)).unwrap();
        // Unreadable and foreign files are skipped, not fatal.
        std::fs::write(dir.join("junk.json"), b"{not json").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::write(dir.join("evil.json"), json!({"id": "../x"}).to_string()).unwrap();
        let ids: Vec<String> = list(&dir).into_iter().map(|c| c.id).collect();
        assert_eq!(ids, ["b2", "c3", "a1"], "newest first");
        let s = &list(&dir)[0];
        assert_eq!((s.title.as_str(), s.n_messages, s.target["run"].as_str()), ("chat b2", 1, Some("worker-pool")));

        // Saving again replaces the file; no temporary file is left behind.
        let mut v = conv("a1", 40);
        v["title"] = "renamed".into();
        save(&dir, &v).unwrap();
        assert_eq!(load(&dir, "a1").unwrap()["title"], "renamed");
        assert_eq!(list(&dir)[0].id, "a1");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");

        assert!(delete(&dir, "b2").unwrap());
        assert!(!delete(&dir, "b2").unwrap(), "already gone");
        assert_eq!(delete_all(&dir).unwrap(), 2);
        assert!(list(&dir).is_empty());
        assert!(dir.join("junk.json").exists(), "delete_all only removes conversations");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A copy made in Explorer keeps the id inside: it must not show as a
    /// second row that cannot be opened, and Delete saved chats must take
    /// it (and a crash's temporary file) off the disk too.
    #[test]
    fn copies_and_renamed_files_are_not_listed_but_are_deleted() {
        let dir = temp("copies");
        save(&dir, &conv("c-a1", 10)).unwrap();
        std::fs::copy(dir.join("c-a1.json"), dir.join("c-a1 - Copy.json")).unwrap();
        std::fs::write(dir.join("c-b2.json"), conv("c-zz", 20).to_string()).unwrap();
        std::fs::write(dir.join("C-A3.json"), conv("c-a3", 5).to_string()).unwrap();
        let stale = dir.join(".c-a1.json.4000000001-0.tmp");
        std::fs::write(&stale, conv("c-a1", 9).to_string()).unwrap();
        let ids: Vec<String> = list(&dir).into_iter().map(|c| c.id).collect();
        assert_eq!(ids, ["c-a1", "c-a3"], "a file is listed under the name load and delete use");
        assert_eq!(load(&dir, "c-a3").unwrap()["id"], "c-a3", "names ignore case, as Windows does");

        std::fs::write(dir.join("notes.json"), json!({ "id": "x", "title": "not a conversation" }).to_string()).unwrap();
        assert_eq!(delete_all(&dir).unwrap(), 4);
        let left: Vec<String> =
            std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(left, ["notes.json"], "only what is not a conversation stays");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ids_cannot_escape_the_folder() {
        let dir = temp("ids");
        for bad in ["", "../x", "a/b", "a\\b", "C:x", "a.json", "a b", &"x".repeat(81)] {
            assert!(!valid_id(bad), "{bad:?}");
            assert!(save(&dir, &conv(bad, 1)).is_err(), "{bad:?}");
            assert!(load(&dir, bad).is_err());
            assert!(delete(&dir, bad).is_err());
        }
        assert!(valid_id("c-1789812345-abc_DEF"));
        assert!(save(&dir, &json!(["not", "an", "object"])).is_err());
        assert!(save(&dir, &json!({"title": "no id"})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn presets_round_trip() {
        let dir = temp("presets");
        let p = dir.join("chat").join("presets.json");
        assert_eq!(load_presets(&p), json!({}));
        let v = json!({ "worker-pool": { "system_prompt": "Be brief.", "params": { "temperature": 0.7 }, "thinking": false } });
        save_presets(&p, &v).unwrap();
        assert_eq!(load_presets(&p), v);
        assert!(save_presets(&p, &json!([1])).is_err());
        std::fs::write(&p, b"[1,2]").unwrap();
        assert_eq!(load_presets(&p), json!({}), "a non-object file reads as empty");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
