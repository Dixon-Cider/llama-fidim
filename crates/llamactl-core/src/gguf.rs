//! Minimal GGUF header reader.
//!
//! Reads only the metadata KV table — never tensor data — so parsing a 60 GB
//! model file costs a few MB of reads. Vendored rather than a crate dependency
//! because the format is stable and we need exactly four things from it:
//! architecture, layer count, quantisation, and the KV-cache-relevant
//! attention metadata (heads, GQA, sliding-window pattern) for the VRAM
//! estimator.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::{Error, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// Refuse strings/arrays larger than this — a corrupt length would otherwise
/// make us allocate absurd buffers.
const MAX_SANE_LEN: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// Arrays are skipped, not stored — we record element type and length so
    /// callers can see what was there (e.g. tokenizer vocab size).
    ArraySkipped {
        elem_type: u32,
        len: u64,
    },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Everything llamactl needs from a model file, plus the raw scalar metadata
/// for display and future estimator refinements.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GgufHeader {
    pub path: PathBuf,
    pub file_size: u64,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub size_label: Option<String>,
    /// `general.file_type` — llama.cpp's quantisation enum (e.g. 15 = Q4_K_M).
    pub file_type: Option<u64>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    /// Per-layer sliding-window size, if the architecture uses SWA.
    pub sliding_window: Option<u64>,
    /// SWA pattern: on e.g. Gemma-family models only 1 in N layers is
    /// full-attention, which changes KV growth with context dramatically.
    pub sliding_window_pattern: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    /// All scalar metadata (arrays recorded as skipped).
    pub metadata: BTreeMap<String, Value>,
}

pub fn read_header(path: &Path) -> Result<GgufHeader> {
    let file = File::open(path).map_err(|e| Error::io(path, e))?;
    let file_size = file.metadata().map_err(|e| Error::io(path, e))?.len();
    let mut r = BufReader::with_capacity(1 << 20, file);

    let mut magic = [0u8; 4];
    read_exact(&mut r, &mut magic, path)?;
    if &magic != GGUF_MAGIC {
        return Err(Error::GgufBadMagic(path.to_path_buf()));
    }
    let gguf_version = read_u32(&mut r, path)?;
    if !(2..=3).contains(&gguf_version) {
        return Err(Error::GgufVersion { path: path.to_path_buf(), version: gguf_version });
    }
    let tensor_count = read_u64(&mut r, path)?;
    let kv_count = read_u64(&mut r, path)?;
    if kv_count > 1_000_000 {
        return Err(malformed(path, format!("implausible kv_count {kv_count}")));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut r, path)?;
        let vtype = read_u32(&mut r, path)?;
        let value = read_value(&mut r, vtype, path)?;
        metadata.insert(key, value);
    }

    let arch = metadata.get("general.architecture").and_then(|v| v.as_str().map(String::from));
    let arch_key = |suffix: &str| -> Option<u64> {
        let a = arch.as_deref()?;
        metadata.get(&format!("{a}.{suffix}")).and_then(|v| v.as_u64())
    };

    Ok(GgufHeader {
        path: path.to_path_buf(),
        file_size,
        gguf_version,
        tensor_count,
        model_name: metadata.get("general.name").and_then(|v| v.as_str().map(String::from)),
        size_label: metadata.get("general.size_label").and_then(|v| v.as_str().map(String::from)),
        file_type: metadata.get("general.file_type").and_then(|v| v.as_u64()),
        block_count: arch_key("block_count"),
        context_length: arch_key("context_length"),
        embedding_length: arch_key("embedding_length"),
        head_count: arch_key("attention.head_count"),
        head_count_kv: arch_key("attention.head_count_kv"),
        sliding_window: arch_key("attention.sliding_window"),
        sliding_window_pattern: arch_key("attention.sliding_window_pattern"),
        expert_count: arch_key("expert_count"),
        expert_used_count: arch_key("expert_used_count"),
        architecture: arch,
        metadata,
    })
}

fn malformed(path: &Path, detail: String) -> Error {
    Error::GgufMalformed { path: path.to_path_buf(), detail }
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8], path: &Path) -> Result<()> {
    r.read_exact(buf).map_err(|e| Error::io(path, e))
}

fn read_u32<R: Read>(r: &mut R, path: &Path) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b, path)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R, path: &Path) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b, path)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R, path: &Path) -> Result<String> {
    let len = read_u64(r, path)?;
    if len > MAX_SANE_LEN {
        return Err(malformed(path, format!("implausible string length {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(r, &mut buf, path)?;
    String::from_utf8(buf).map_err(|e| malformed(path, format!("non-UTF8 string: {e}")))
}

/// Fixed byte width of a scalar GGUF value type, if it has one.
fn scalar_width(vtype: u32) -> Option<u64> {
    match vtype {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4 | 5 | 6 => Some(4), // u32, i32, f32
        10 | 11 | 12 => Some(8), // u64, i64, f64
        _ => None,
    }
}

fn read_value<R: Read + Seek>(r: &mut R, vtype: u32, path: &Path) -> Result<Value> {
    Ok(match vtype {
        0 => Value::U64(read_byte(r, path)? as u64),
        1 => Value::I64(read_byte(r, path)? as i8 as i64),
        2 => {
            let mut b = [0u8; 2];
            read_exact(r, &mut b, path)?;
            Value::U64(u16::from_le_bytes(b) as u64)
        }
        3 => {
            let mut b = [0u8; 2];
            read_exact(r, &mut b, path)?;
            Value::I64(i16::from_le_bytes(b) as i64)
        }
        4 => Value::U64(read_u32(r, path)? as u64),
        5 => Value::I64(read_u32(r, path)? as i32 as i64),
        6 => {
            let mut b = [0u8; 4];
            read_exact(r, &mut b, path)?;
            Value::F64(f32::from_le_bytes(b) as f64)
        }
        7 => Value::Bool(read_byte(r, path)? != 0),
        8 => Value::Str(read_string(r, path)?),
        9 => {
            // Array: elem type + count. Seek past fixed-width elements; walk
            // string/nested arrays element-wise (tokenizer vocabs are string
            // arrays — walking lengths is still only a few MB of I/O).
            let elem_type = read_u32(r, path)?;
            let len = read_u64(r, path)?;
            if len > MAX_SANE_LEN {
                return Err(malformed(path, format!("implausible array length {len}")));
            }
            if let Some(w) = scalar_width(elem_type) {
                let bytes = w.checked_mul(len)
                    .ok_or_else(|| malformed(path, "array size overflow".into()))?;
                r.seek(SeekFrom::Current(bytes as i64)).map_err(|e| Error::io(path, e))?;
            } else if elem_type == 8 {
                for _ in 0..len {
                    let slen = read_u64(r, path)?;
                    if slen > MAX_SANE_LEN {
                        return Err(malformed(path, format!("implausible string length {slen}")));
                    }
                    r.seek(SeekFrom::Current(slen as i64)).map_err(|e| Error::io(path, e))?;
                }
            } else if elem_type == 9 {
                // Nested arrays are legal in the format but unseen in real
                // model files; walking them without a use case is dead code.
                return Err(malformed(path, "nested arrays not supported".into()));
            } else {
                return Err(malformed(path, format!("unknown array element type {elem_type}")));
            }
            Value::ArraySkipped { elem_type, len }
        }
        10 => Value::U64(read_u64(r, path)?),
        11 => Value::I64(read_u64(r, path)? as i64),
        12 => {
            let mut b = [0u8; 8];
            read_exact(r, &mut b, path)?;
            Value::F64(f64::from_le_bytes(b))
        }
        other => return Err(malformed(path, format!("unknown value type {other}"))),
    })
}

fn read_byte<R: Read>(r: &mut R, path: &Path) -> Result<u8> {
    let mut b = [0u8; 1];
    read_exact(r, &mut b, path)?;
    Ok(b[0])
}

/// Human name for llama.cpp's `general.file_type` enum (the common ones).
pub fn file_type_name(ft: u64) -> String {
    match ft {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        7 => "Q8_0".into(),
        8 => "Q5_0".into(),
        9 => "Q5_1".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K_S".into(),
        12 => "Q3_K_M".into(),
        13 => "Q3_K_L".into(),
        14 => "Q4_K_S".into(),
        15 => "Q4_K_M".into(),
        16 => "Q5_K_S".into(),
        17 => "Q5_K_M".into(),
        18 => "Q6_K".into(),
        19 => "IQ2_XXS".into(),
        24 => "IQ1_S".into(),
        30 => "BF16".into(),
        other => format!("file_type {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Build a minimal synthetic GGUF header in memory.
    fn synth_gguf(kvs: &[(&str, SynthVal)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            match v {
                SynthVal::U32(x) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                SynthVal::Str(s) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
                SynthVal::StrArray(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&8u32.to_le_bytes()); // elem type: string
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in items {
                        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                        out.extend_from_slice(s.as_bytes());
                    }
                }
                SynthVal::F32Array(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&6u32.to_le_bytes()); // elem type: f32
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for f in items {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                }
            }
        }
        out
    }

    enum SynthVal {
        U32(u32),
        Str(&'static str),
        StrArray(Vec<&'static str>),
        F32Array(Vec<f32>),
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("llamactl-gguf-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}-{}.gguf", std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn parses_architecture_and_swa_metadata() {
        let bytes = synth_gguf(&[
            ("general.architecture", SynthVal::Str("gemma4")),
            ("general.name", SynthVal::Str("Test Model")),
            ("gemma4.block_count", SynthVal::U32(30)),
            ("gemma4.context_length", SynthVal::U32(262144)),
            ("gemma4.attention.head_count", SynthVal::U32(16)),
            ("gemma4.attention.head_count_kv", SynthVal::U32(4)),
            ("gemma4.attention.sliding_window", SynthVal::U32(1024)),
            ("gemma4.attention.sliding_window_pattern", SynthVal::U32(6)),
            ("tokenizer.ggml.tokens", SynthVal::StrArray(vec!["a", "bb", "ccc"])),
            ("some.floats", SynthVal::F32Array(vec![1.0, 2.0])),
        ]);
        let path = write_temp("swa", &bytes);
        let h = read_header(&path).unwrap();
        assert_eq!(h.architecture.as_deref(), Some("gemma4"));
        assert_eq!(h.block_count, Some(30));
        assert_eq!(h.head_count_kv, Some(4));
        assert_eq!(h.sliding_window, Some(1024));
        assert_eq!(h.sliding_window_pattern, Some(6));
        // Arrays skipped but recorded.
        assert_eq!(
            h.metadata.get("tokenizer.ggml.tokens"),
            Some(&Value::ArraySkipped { elem_type: 8, len: 3 })
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_bad_magic() {
        let path = write_temp("badmagic", b"NOPE1234");
        assert!(matches!(read_header(&path), Err(Error::GgufBadMagic(_))));
        std::fs::remove_file(path).ok();
    }

    /// If real models are present on this machine, parse them all — the
    /// strongest fixture is the actual inventory.
    #[test]
    fn parses_real_models_when_present() {
        let root = std::path::Path::new("E:/models");
        if !root.exists() {
            return;
        }
        let mut checked = 0;
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) {
                    let h = read_header(&p).unwrap_or_else(|err| panic!("failed on {p:?}: {err}"));
                    assert!(h.architecture.is_some(), "no architecture in {p:?}");
                    checked += 1;
                }
            }
        }
        eprintln!("parsed {checked} real GGUF files");
    }
}
