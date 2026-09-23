//! Minimal GGUF header reader.
//!
//! Reads the metadata KV table and the tensor-info table — never tensor
//! data — so parsing a 60 GB model file costs a few MB of reads. Vendored
//! rather than a crate dependency because the format is stable and we need
//! a handful of things from it: architecture, layer count, quantisation, the
//! KV-cache-relevant attention metadata (heads, GQA, sliding-window pattern)
//! for the VRAM estimator, and the tokenizer and tensor types a build must
//! know to load the file.
//!
//! The parser runs over any `Read + Seek`, so a Range-fetched prefix of a
//! remote file parses the same way as a local file. When the bytes run out
//! it says how many it needed (`Error::GgufTruncated`), and
//! `ReadMode::UntilTokenizer` stops before the tokenizer arrays, which hold
//! most of a header's megabytes: the hyperparameters and the tokenizer's
//! pre-tokenizer name sit in the first 1-2 KB.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::{Error, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// Refuse strings/arrays larger than this — a corrupt length would otherwise
/// make us allocate absurd buffers.
const MAX_SANE_LEN: u64 = 256 * 1024 * 1024;
/// Tensor names are at most 63 bytes in ggml (GGML_MAX_NAME); anything far
/// past that is a corrupt length, not a name.
const MAX_TENSOR_NAME: u64 = 64 * 1024;
/// ggml tensors have at most 4 dimensions (GGML_MAX_DIMS).
const MAX_TENSOR_DIMS: u32 = 8;
/// More tensors than any real model has (the 375B MoEs carry ~3,000).
const MAX_TENSOR_COUNT: u64 = 10_000_000;

/// How much of a header to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    /// Every KV pair and the tensor-info table.
    Full,
    /// Stop at the first large tokenizer array (the vocabulary), recording
    /// its length as `vocab_size`. Everything before it is parsed: the
    /// general and architecture keys and `tokenizer.ggml.model` / `.pre`
    /// (which converters write before the vocabulary). Keys that
    /// converters write after the tokenizer (`general.file_type` from
    /// llama-quantize, `split.*` from gguf-split) are not seen, and there are
    /// no tensors.
    UntilTokenizer,
}

/// One entry of the tensor-info table.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TensorInfo {
    pub name: String,
    /// `enum ggml_type` id (see `ggml_type_name`).
    pub ggml_type: u32,
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// Small scalar arrays are retained — per-layer attention metadata
    /// (kv-head counts, SWA flags) lives in these on modern architectures.
    Array(Vec<Value>),
    /// Large or string arrays are skipped, recording element type and length
    /// so callers can see what was there (e.g. tokenizer vocab size).
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
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// Any numeric scalar as f64 (the converter writes sampling defaults as f32).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::U64(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    /// Array of unsigned integers, if every element converts.
    pub fn as_u64_array(&self) -> Option<Vec<u64>> {
        match self {
            Value::Array(items) => items.iter().map(|v| v.as_u64()).collect(),
            _ => None,
        }
    }
    /// Array of bools, if every element is one.
    pub fn as_bool_array(&self) -> Option<Vec<bool>> {
        match self {
            Value::Array(items) => items.iter().map(|v| v.as_bool()).collect(),
            _ => None,
        }
    }
}

/// Everything Llama FIDIM needs from a model file, plus the raw scalar metadata
/// for display and future estimator refinements.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    /// Hugging Face repo of the ORIGINAL model (`general.base_model.0.*`),
    /// where the creator's generation_config.json lives; falls back to the
    /// quant repo (`general.source.*` / `general.repo_url`).
    pub source_repo: Option<String>,
    /// The repo this GGUF itself came from (`general.source.*`), when it
    /// differs from the base model — the quantizer's repo.
    pub quant_repo: Option<String>,
    /// Creator sampling defaults embedded by the converter (`general.sampling.*`
    /// from the source repo's generation_config.json). None = not present.
    pub sampling_temp: Option<f64>,
    pub sampling_top_k: Option<u32>,
    pub sampling_top_p: Option<f64>,
    pub sampling_min_p: Option<f64>,
    pub sampling_repeat_penalty: Option<f64>,
    /// `<arch>.nextn_predict_layers`: the model ships a built-in Multi-Token
    /// Prediction head (Qwen 3.5+, DeepSeek V3). > 0 = `--spec-type draft-mtp`
    /// works without a separate draft file.
    pub nextn_predict_layers: Option<u64>,
    /// Hybrid linear/full attention (`<arch>.full_attention_interval`, e.g.
    /// Qwen 3.5+/3.8 = 4): only every Nth layer keeps a KV cache; the others
    /// are Gated-DeltaNet/SSM layers with a small fixed recurrent state.
    pub full_attention_interval: Option<u64>,
    /// Recurrent-state geometry for those layers (`<arch>.ssm.*`).
    pub ssm_inner_size: Option<u64>,
    pub ssm_state_size: Option<u64>,
    pub ssm_conv_kernel: Option<u64>,
    pub ssm_group_count: Option<u64>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    /// Scalar KV-head count (uniform architectures).
    pub head_count_kv: Option<u64>,
    /// Per-layer KV-head counts (e.g. Gemma-4 stores an i32[block_count]).
    pub head_count_kv_per_layer: Option<Vec<u64>>,
    /// Attention head dims: `attention.key_length` / `value_length`, with
    /// `_swa` variants for sliding-window layers where present. These are the
    /// authoritative head dims — NOT embedding/head_count.
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub key_length_swa: Option<u64>,
    pub value_length_swa: Option<u64>,
    /// Per-layer sliding-window size, if the architecture uses SWA.
    pub sliding_window: Option<u64>,
    /// Scalar SWA pattern (older style): every Nth layer is full-attention.
    pub sliding_window_pattern: Option<u64>,
    /// Per-layer SWA flags (newer style, bool[block_count]): true = sliding.
    pub swa_layer_flags: Option<Vec<bool>>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    /// `diffusion.canvas_length`: tokens denoised per block by a diffusion
    /// LM. The key is literal, not arch-prefixed; the DiffusionGemma runner
    /// reads it the same way and refuses a model without it.
    #[serde(default)]
    pub diffusion_canvas_length: Option<u64>,
    /// `<arch>.attention.causal`; false on diffusion LMs.
    #[serde(default)]
    pub attention_causal: Option<bool>,
    /// Length of `tokenizer.ggml.tokens`.
    #[serde(default)]
    pub vocab_size: Option<u64>,
    /// `tokenizer.ggml.model`, e.g. `gpt2` (BPE) or `llama` (SentencePiece).
    #[serde(default)]
    pub tokenizer_model: Option<String>,
    /// `tokenizer.ggml.pre`: the pre-tokenizer a build must know by name
    /// (e.g. `k2-horizon`), or it refuses to load the vocabulary.
    #[serde(default)]
    pub tokenizer_pre: Option<String>,
    /// `<arch>.rope.scaling.type`, e.g. `yarn`.
    #[serde(default)]
    pub rope_scaling_type: Option<String>,
    /// `split.no` / `split.count` of a gguf-split shard (0-based number).
    #[serde(default)]
    pub split_no: Option<u16>,
    #[serde(default)]
    pub split_count: Option<u16>,
    /// The tensor-info table (`ReadMode::Full` only): this file's, or once
    /// `fold_split_shards` has run, every shard's. Not serialized: a model
    /// has hundreds to thousands of entries, and every scan result carries
    /// its header. `max_tensor_type` is the summary callers need.
    #[serde(skip)]
    pub tensors: Vec<TensorInfo>,
    /// Highest ggml type id among the tensors, kept beside the table so it
    /// survives serialization. None in `UntilTokenizer` mode or with no
    /// tensors.
    #[serde(default)]
    pub max_tensor_type_id: Option<u32>,
    /// The table holds one shard of a split set (`split.count` > 1): each
    /// gguf-split shard lists only its own tensors, and a first shard
    /// written with `--no-tensor-first-split` lists none, so the model's
    /// tensor types are not known until `fold_split_shards` adds the rest.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tensor_types_partial: bool,
    /// Parsed in `ReadMode::UntilTokenizer`: the keys after the tokenizer
    /// and the tensor table are missing by design.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub partial: bool,
    /// All scalar metadata (arrays recorded as skipped).
    pub metadata: BTreeMap<String, Value>,
}

/// Architectures that are diffusion LMs: llama-server cannot serve them.
pub const DIFFUSION_ARCHES: &[&str] = &["diffusion-gemma", "dream", "llada", "llada-moe", "rnd1"];

impl GgufHeader {
    pub fn is_diffusion(&self) -> bool {
        self.diffusion_canvas_length.is_some()
            || self.architecture.as_deref().is_some_and(|a| DIFFUSION_ARCHES.contains(&a))
    }

    /// Whether the DiffusionGemma runner can load this file: it handles only
    /// its own architecture and exits on a model without the canvas key.
    pub fn runner_supported(&self) -> bool {
        self.architecture.as_deref() == Some("diffusion-gemma") && self.diffusion_canvas_length.is_some()
    }

    /// The engine a profile for this model needs.
    pub fn engine(&self) -> crate::profile::Engine {
        if self.is_diffusion() {
            crate::profile::Engine::DiffusionGemma
        } else {
            crate::profile::Engine::LlamaServer
        }
    }

    /// Highest ggml tensor type id in the model: a build whose ggml has
    /// fewer types (GGML_TYPE_COUNT) cannot load it. None = unknown: the
    /// tensor table was not read (`ReadMode::UntilTokenizer`), is empty, or
    /// is one shard's of a split set whose other shards were not folded in
    /// (`tensor_types_partial`).
    pub fn max_tensor_type(&self) -> Option<u32> {
        if self.tensor_types_partial {
            return None;
        }
        self.tensors.iter().map(|t| t.ggml_type).max().max(self.max_tensor_type_id)
    }

    /// Fold the rest of a split set into this header of its first shard:
    /// `rest` is every other shard's header (read in `ReadMode::Full`), in
    /// order. Their tensor tables join this one, and `tensor_count`,
    /// `file_size` and `max_tensor_type` become the whole model's. The
    /// types stay unknown if a shard is missing or was read only up to its
    /// tokenizer.
    pub fn fold_split_shards(&mut self, rest: Vec<GgufHeader>) {
        let complete = !self.partial
            && !rest.iter().any(|s| s.partial)
            && self.split_count.is_none_or(|n| usize::from(n) == rest.len() + 1);
        for s in rest {
            self.file_size += s.file_size;
            self.tensor_count += s.tensor_count;
            self.max_tensor_type_id = self.max_tensor_type_id.max(s.max_tensor_type_id);
            self.tensors.extend(s.tensors);
        }
        self.tensor_types_partial = !complete;
    }
}

/// Parse a local GGUF file's header in full (KV table and tensor infos).
pub fn read_header(path: &Path) -> Result<GgufHeader> {
    let file = File::open(path).map_err(|e| Error::io(path, e))?;
    let file_size = file.metadata().map_err(|e| Error::io(path, e))?.len();
    let mut r = BufReader::with_capacity(1 << 20, file);
    read_header_from(&mut r, file_size, path, ReadMode::Full)
}

/// Parse a GGUF header from `r`, positioned at the start of the file.
/// `file_size` is the size of the whole file (not of what `r` holds) and
/// `label` names it in errors and in `GgufHeader::path` (a local path, or a
/// pseudo-path such as `hf://owner/repo@sha/file.gguf`).
///
/// When `r` ends early the error is `Error::GgufTruncated` with the offset
/// the parser needed, so a caller holding a prefix of the file can fetch
/// more and try again.
pub fn read_header_from<R: Read + Seek>(r: &mut R, file_size: u64, label: &Path, mode: ReadMode) -> Result<GgufHeader> {
    let base = r.stream_position().map_err(|e| Error::io(label, e))?;
    let mut src = Src { r, pos: base, path: label };

    let mut magic = [0u8; 4];
    src.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(Error::GgufBadMagic(label.to_path_buf()));
    }
    let gguf_version = src.read_u32()?;
    if !(2..=3).contains(&gguf_version) {
        return Err(Error::GgufVersion { path: label.to_path_buf(), version: gguf_version });
    }
    let tensor_count = src.read_u64()?;
    let kv_count = src.read_u64()?;
    if kv_count > 1_000_000 {
        return Err(src.malformed(format!("implausible kv_count {kv_count}")));
    }

    let mut metadata = BTreeMap::new();
    let mut partial = false;
    for _ in 0..kv_count {
        let key = src.read_string()?;
        let vtype = src.read_u32()?;
        if mode == ReadMode::UntilTokenizer && vtype == 9 && key.starts_with("tokenizer.") {
            // The vocabulary (and merges, scores, types) are megabytes of
            // arrays; their headers alone say how long they are. Small ones
            // (Gemma 4's suppress_tokens, before the model name) are read.
            let elem_type = src.read_u32()?;
            let len = src.read_u64()?;
            if !array_is_kept(elem_type, len) {
                metadata.insert(key, Value::ArraySkipped { elem_type, len });
                partial = true;
                break;
            }
            let value = read_array(&mut src, elem_type, len)?;
            metadata.insert(key, value);
            continue;
        }
        let value = read_value(&mut src, vtype)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::new();
    if mode == ReadMode::Full {
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(src.malformed(format!("implausible tensor_count {tensor_count}")));
        }
        tensors.reserve(tensor_count.min(65_536) as usize);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_info(&mut src)?);
        }
    } else {
        partial = true;
    }

    let arch = metadata.get("general.architecture").and_then(|v| v.as_str().map(String::from));
    let arch_key = |suffix: &str| -> Option<u64> {
        let a = arch.as_deref()?;
        metadata.get(&format!("{a}.{suffix}")).and_then(|v| v.as_u64())
    };
    let arch_val = |suffix: &str| -> Option<&Value> {
        let a = arch.as_deref()?;
        metadata.get(&format!("{a}.{suffix}"))
    };
    let string = |key: &str| metadata.get(key).and_then(|v| v.as_str().map(String::from));
    let array_len = |key: &str| {
        metadata.get(key).and_then(|v| match v {
            Value::ArraySkipped { len, .. } => Some(*len),
            Value::Array(items) => Some(items.len() as u64),
            _ => None,
        })
    };
    let split_u16 = |key: &str| metadata.get(key).and_then(Value::as_u64).and_then(|v| u16::try_from(v).ok());
    let split_count = split_u16("split.count");

    Ok(GgufHeader {
        path: label.to_path_buf(),
        file_size,
        gguf_version,
        tensor_count,
        model_name: string("general.name"),
        size_label: string("general.size_label"),
        file_type: metadata.get("general.file_type").and_then(|v| v.as_u64()),
        source_repo: source_repo_from(&metadata),
        quant_repo: quant_repo_from(&metadata),
        // Stored as f32 by the converter; round away the f32->f64 noise
        // (0.949999988 -> 0.95) so the UI shows what the creator wrote.
        sampling_temp: metadata.get("general.sampling.temp").and_then(|v| v.as_f64()).map(round4),
        sampling_top_k: metadata.get("general.sampling.top_k").and_then(|v| v.as_f64()).map(|k| k.round() as u32),
        sampling_top_p: metadata.get("general.sampling.top_p").and_then(|v| v.as_f64()).map(round4),
        sampling_min_p: metadata.get("general.sampling.min_p").and_then(|v| v.as_f64()).map(round4),
        sampling_repeat_penalty: metadata
            .get("general.sampling.repeat_penalty")
            .or_else(|| metadata.get("general.sampling.repetition_penalty"))
            .and_then(|v| v.as_f64())
            .map(round4),
        nextn_predict_layers: arch_key("nextn_predict_layers").or_else(|| {
            metadata.iter().find(|(k, _)| k.ends_with(".nextn_predict_layers")).and_then(|(_, v)| v.as_u64())
        }),
        full_attention_interval: arch_key("full_attention_interval"),
        ssm_inner_size: arch_key("ssm.inner_size"),
        ssm_state_size: arch_key("ssm.state_size"),
        ssm_conv_kernel: arch_key("ssm.conv_kernel"),
        ssm_group_count: arch_key("ssm.group_count"),
        block_count: arch_key("block_count"),
        context_length: arch_key("context_length"),
        embedding_length: arch_key("embedding_length"),
        head_count: arch_key("attention.head_count"),
        head_count_kv: arch_key("attention.head_count_kv"),
        head_count_kv_per_layer: arch_val("attention.head_count_kv")
            .and_then(|v| v.as_u64_array()),
        key_length: arch_key("attention.key_length"),
        value_length: arch_key("attention.value_length"),
        key_length_swa: arch_key("attention.key_length_swa"),
        value_length_swa: arch_key("attention.value_length_swa"),
        sliding_window: arch_key("attention.sliding_window"),
        sliding_window_pattern: arch_key("attention.sliding_window_pattern"),
        swa_layer_flags: arch_val("attention.sliding_window_pattern")
            .and_then(|v| v.as_bool_array()),
        expert_count: arch_key("expert_count"),
        expert_used_count: arch_key("expert_used_count"),
        diffusion_canvas_length: metadata.get("diffusion.canvas_length").and_then(Value::as_u64),
        attention_causal: arch_val("attention.causal").and_then(|v| v.as_bool()),
        // The token list, or in an early-stopped read whichever per-token
        // array came first (scores and types are as long as the list).
        vocab_size: array_len("tokenizer.ggml.tokens")
            .or_else(|| array_len("tokenizer.ggml.scores"))
            .or_else(|| array_len("tokenizer.ggml.token_type")),
        tokenizer_model: string("tokenizer.ggml.model"),
        tokenizer_pre: string("tokenizer.ggml.pre"),
        rope_scaling_type: arch_val("rope.scaling.type").and_then(|v| v.as_str().map(String::from)),
        split_no: split_u16("split.no"),
        split_count,
        max_tensor_type_id: tensors.iter().map(|t| t.ggml_type).max(),
        tensor_types_partial: mode == ReadMode::Full && split_count.is_some_and(|n| n > 1),
        tensors,
        partial,
        architecture: arch,
        metadata,
    })
}

/// The reader plus its absolute offset, so a short read can say how many
/// bytes the parse needed. Counted here rather than asked of the reader:
/// `stream_position` on a file is a syscall, and a vocabulary walk makes
/// hundreds of thousands of reads.
struct Src<'a, R> {
    r: &'a mut R,
    pos: u64,
    path: &'a Path,
}

impl<R: Read + Seek> Src<'_, R> {
    fn malformed(&self, detail: String) -> Error {
        malformed(self.path, detail)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        match self.r.read_exact(buf) {
            Ok(()) => {
                self.pos += buf.len() as u64;
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                Err(Error::GgufTruncated { path: self.path.to_path_buf(), at: self.pos + buf.len() as u64 })
            }
            Err(e) => Err(Error::io(self.path, e)),
        }
    }

    fn read_byte(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_string(&mut self) -> Result<String> {
        self.read_string_max(MAX_SANE_LEN)
    }

    fn read_string_max(&mut self, max: u64) -> Result<String> {
        let len = self.read_u64()?;
        if len > max {
            return Err(self.malformed(format!("implausible string length {len}")));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| self.malformed(format!("non-UTF8 string: {e}")))
    }

    /// Skip `n` bytes. Short runs are consumed through the reader's buffer
    /// (no syscall); only genuinely large runs pay for a seek. A seek past
    /// the end succeeds; the next read reports the truncation.
    fn skip(&mut self, n: u64) -> Result<()> {
        const THROUGH_BUFFER: u64 = 64 * 1024;
        if n <= THROUGH_BUFFER {
            let mut left = n;
            let mut scratch = [0u8; 4096];
            while left > 0 {
                let take = left.min(scratch.len() as u64) as usize;
                self.read_exact(&mut scratch[..take])?;
                left -= take as u64;
            }
            Ok(())
        } else {
            let step = i64::try_from(n).map_err(|_| self.malformed(format!("implausible skip of {n} bytes")))?;
            self.r.seek(SeekFrom::Current(step)).map_err(|e| Error::io(self.path, e))?;
            self.pos += n;
            Ok(())
        }
    }
}

fn malformed(path: &Path, detail: String) -> Error {
    Error::GgufMalformed { path: path.to_path_buf(), detail }
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

fn read_value<R: Read + Seek>(src: &mut Src<'_, R>, vtype: u32) -> Result<Value> {
    Ok(match vtype {
        0 => Value::U64(src.read_byte()? as u64),
        1 => Value::I64(src.read_byte()? as i8 as i64),
        2 => {
            let mut b = [0u8; 2];
            src.read_exact(&mut b)?;
            Value::U64(u16::from_le_bytes(b) as u64)
        }
        3 => {
            let mut b = [0u8; 2];
            src.read_exact(&mut b)?;
            Value::I64(i16::from_le_bytes(b) as i64)
        }
        4 => Value::U64(src.read_u32()? as u64),
        5 => Value::I64(src.read_u32()? as i32 as i64),
        6 => {
            let mut b = [0u8; 4];
            src.read_exact(&mut b)?;
            Value::F64(f32::from_le_bytes(b) as f64)
        }
        7 => Value::Bool(src.read_byte()? != 0),
        8 => Value::Str(src.read_string()?),
        9 => {
            let elem_type = src.read_u32()?;
            let len = src.read_u64()?;
            read_array(src, elem_type, len)?
        }
        10 => Value::U64(src.read_u64()?),
        11 => Value::I64(src.read_u64()? as i64),
        12 => {
            let mut b = [0u8; 8];
            src.read_exact(&mut b)?;
            Value::F64(f64::from_le_bytes(b))
        }
        other => return Err(src.malformed(format!("unknown value type {other}"))),
    })
}

/// Arrays small enough to keep: scalar ones of at most 4096 elements
/// (per-layer attention metadata). Longer ones and string arrays are skipped.
fn array_is_kept(elem_type: u32, len: u64) -> bool {
    scalar_width(elem_type).is_some() && len <= 4096
}

/// An array's elements, after its element type and count. Small scalar
/// arrays are retained (per-layer attention metadata). Large scalar arrays
/// are seeked past; string arrays are walked element-wise (tokenizer vocabs
/// are string arrays — walking lengths is still only MBs of I/O).
fn read_array<R: Read + Seek>(src: &mut Src<'_, R>, elem_type: u32, len: u64) -> Result<Value> {
    if len > MAX_SANE_LEN {
        return Err(src.malformed(format!("implausible array length {len}")));
    }
    if array_is_kept(elem_type, len) {
        let mut items = Vec::with_capacity(len as usize);
        for _ in 0..len {
            items.push(read_value(src, elem_type)?);
        }
        return Ok(Value::Array(items));
    }
    if let Some(w) = scalar_width(elem_type) {
        let bytes = w.checked_mul(len).ok_or_else(|| src.malformed("array size overflow".into()))?;
        src.skip(bytes)?;
    } else if elem_type == 8 {
        // Tokenizer vocab + merges: ~262K short strings each. A
        // `seek` per string discards the BufReader buffer and costs a
        // syscall + refill every time (measured 15-25 s per model);
        // consuming the bytes through the buffer instead is ~ms.
        for _ in 0..len {
            let slen = src.read_u64()?;
            if slen > MAX_SANE_LEN {
                return Err(src.malformed(format!("implausible string length {slen}")));
            }
            src.skip(slen)?;
        }
    } else if elem_type == 9 {
        // Nested arrays are legal in the format but unseen in real
        // model files; walking them without a use case is dead code.
        return Err(src.malformed("nested arrays not supported".into()));
    } else {
        return Err(src.malformed(format!("unknown array element type {elem_type}")));
    }
    Ok(Value::ArraySkipped { elem_type, len })
}

/// One tensor-info entry: name, dims, type, data offset (not kept).
fn read_tensor_info<R: Read + Seek>(src: &mut Src<'_, R>) -> Result<TensorInfo> {
    let name = src.read_string_max(MAX_TENSOR_NAME)?;
    let n_dims = src.read_u32()?;
    if n_dims > MAX_TENSOR_DIMS {
        return Err(src.malformed(format!("tensor {name}: implausible dimension count {n_dims}")));
    }
    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        dims.push(src.read_u64()?);
    }
    let ggml_type = src.read_u32()?;
    let _offset = src.read_u64()?;
    Ok(TensorInfo { name, ggml_type, dims })
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// `owner/name` of the model's Hugging Face repo from the GGUF `general.*`
/// provenance keys, preferring the base (creator) model over the quant repo.
/// Accepts either `repo_url` (`https://huggingface.co/owner/name`) or the
/// `organization` + `name` pair the converter writes.
pub fn source_repo_from(metadata: &BTreeMap<String, Value>) -> Option<String> {
    repo_from_keys(metadata, "general.base_model.0.repo_url", "general.base_model.0.organization", "general.base_model.0.name")
        .or_else(|| quant_repo_from(metadata))
}

/// The quantizer's own repo (`general.source.*`, or the top-level
/// `general.repo_url` / `general.organization` + `general.name` pair).
pub fn quant_repo_from(metadata: &BTreeMap<String, Value>) -> Option<String> {
    repo_from_keys(metadata, "general.source.repo_url", "general.source.organization", "general.source.name")
        .or_else(|| repo_from_keys(metadata, "general.source.url", "", ""))
        .or_else(|| repo_from_keys(metadata, "general.repo_url", "general.organization", "general.name"))
}

fn repo_from_keys(
    metadata: &BTreeMap<String, Value>,
    url_key: &str,
    org_key: &str,
    name_key: &str,
) -> Option<String> {
    let s = |k: &str| {
        metadata.get(k).and_then(|v| v.as_str()).map(|x| x.trim().to_string()).filter(|x| !x.is_empty())
    };
    if let Some(u) = s(url_key) {
        if let Some(rest) = u.split("huggingface.co/").nth(1) {
            let parts: Vec<&str> = rest.trim_matches('/').split('/').collect();
            if parts.len() >= 2 {
                return Some(format!("{}/{}", parts[0], parts[1]));
            }
        }
    }
    if org_key.is_empty() {
        return None;
    }
    let (o, n) = (s(org_key)?, s(name_key)?);
    Some(format!("{}/{}", o.replace(' ', "-"), n.replace(' ', "-")))
}

/// Human name for llama.cpp's `general.file_type` enum (`llama_ftype` in
/// llama.h). Retired ids (4-6, 33-35) and ids newer than this table read as
/// `file_type N`.
pub fn file_type_name(ft: u64) -> String {
    let name = match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        other => return format!("file_type {other}"),
    };
    name.into()
}

/// Human name for a tensor's `enum ggml_type` id (ggml.h). Ids past the
/// table (a fork's own types, e.g. ROCmFPX's 100-119) read as `type N`.
pub fn ggml_type_name(t: u32) -> String {
    let name = match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        other => return format!("type {other}"),
    };
    name.into()
}

/// A gguf-split shard name, `<prefix>-NNNNN-of-MMMMM.gguf` (llama.cpp's
/// `llama_split_path` format): `(prefix, number, count)`, number 1-based.
/// Case-insensitive on the extension; None for anything else, including a
/// number of 0 or past the count.
pub fn split_name(file_name: &str) -> Option<(&str, u32, u32)> {
    let stem = file_name.len().checked_sub(5).filter(|&i| file_name.is_char_boundary(i)).and_then(|i| {
        file_name[i..].eq_ignore_ascii_case(".gguf").then(|| &file_name[..i])
    })?;
    // "-00001-of-00003" is 15 bytes, all ASCII when it matches.
    let cut = stem.len().checked_sub(15).filter(|&i| stem.is_char_boundary(i))?;
    let (prefix, tail) = stem.split_at(cut);
    let b = tail.as_bytes();
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    if b[0] != b'-' || !digits(1..6) || !b[6..10].eq_ignore_ascii_case(b"-of-") || !digits(10..15) || prefix.is_empty() {
        return None;
    }
    let no: u32 = tail[1..6].parse().ok()?;
    let count: u32 = tail[10..15].parse().ok()?;
    (no >= 1 && no <= count).then_some((prefix, no, count))
}

/// GGUF builders shared by the tests of the modules that read headers.
#[cfg(test)]
pub(crate) mod testing {
    /// A gguf-split shard: `general.architecture` on the first only, the
    /// `split.*` keys, and this shard's own tensor-info table of
    /// (name, ggml type).
    pub fn split_shard(arch: Option<&str>, no: u16, count: u16, tensors: &[(&str, u32)]) -> Vec<u8> {
        fn key(out: &mut Vec<u8>, k: &str) {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(2 + arch.is_some() as u64).to_le_bytes());
        if let Some(arch) = arch {
            key(&mut out, "general.architecture");
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(arch.len() as u64).to_le_bytes());
            out.extend_from_slice(arch.as_bytes());
        }
        for (k, v) in [("split.no", no), ("split.count", count)] {
            key(&mut out, k);
            out.extend_from_slice(&2u32.to_le_bytes()); // u16
            out.extend_from_slice(&v.to_le_bytes());
        }
        for (name, ggml_type) in tensors {
            key(&mut out, name);
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&64u64.to_le_bytes());
            out.extend_from_slice(&ggml_type.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Build a minimal synthetic GGUF header in memory.
    fn synth_gguf(kvs: &[(&str, SynthVal)]) -> Vec<u8> {
        synth_gguf_with_tensors(kvs, &[])
    }

    /// The same, followed by a tensor-info table of (name, ggml type, dims).
    fn synth_gguf_with_tensors(kvs: &[(&str, SynthVal)], tensors: &[(&str, u32, &[u64])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            match v {
                SynthVal::U32(x) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                SynthVal::U16(x) => {
                    out.extend_from_slice(&2u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                SynthVal::Bool(b) => {
                    out.extend_from_slice(&7u32.to_le_bytes());
                    out.push(if *b { 1 } else { 0 });
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
                SynthVal::I32Array(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&5u32.to_le_bytes()); // elem type: i32
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for v in items {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                SynthVal::BoolArray(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&7u32.to_le_bytes()); // elem type: bool
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for v in items {
                        out.push(if *v { 1 } else { 0 });
                    }
                }
            }
        }
        for (name, ggml_type, dims) in tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ggml_type.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes()); // data offset
        }
        out
    }

    enum SynthVal {
        U32(u32),
        U16(u16),
        Bool(bool),
        Str(&'static str),
        StrArray(Vec<&'static str>),
        F32Array(Vec<f32>),
        I32Array(Vec<i32>),
        BoolArray(Vec<bool>),
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("fidim-gguf-tests");
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
        // String arrays skipped but recorded.
        assert_eq!(
            h.metadata.get("tokenizer.ggml.tokens"),
            Some(&Value::ArraySkipped { elem_type: 8, len: 3 })
        );
        std::fs::remove_file(path).ok();
    }

    /// The Gemma-4 style: per-layer kv-head counts (i32[]) and SWA flags
    /// (bool[]), plus explicit key/value lengths with SWA variants.
    #[test]
    fn parses_per_layer_attention_arrays() {
        let bytes = synth_gguf(&[
            ("general.architecture", SynthVal::Str("gemma4")),
            ("gemma4.block_count", SynthVal::U32(4)),
            ("gemma4.attention.head_count", SynthVal::U32(16)),
            ("gemma4.attention.head_count_kv", SynthVal::I32Array(vec![2, 2, 2, 4])),
            ("gemma4.attention.key_length", SynthVal::U32(512)),
            ("gemma4.attention.value_length", SynthVal::U32(512)),
            ("gemma4.attention.key_length_swa", SynthVal::U32(256)),
            ("gemma4.attention.value_length_swa", SynthVal::U32(256)),
            ("gemma4.attention.sliding_window", SynthVal::U32(1024)),
            (
                "gemma4.attention.sliding_window_pattern",
                SynthVal::BoolArray(vec![true, true, true, false]),
            ),
        ]);
        let path = write_temp("perlayer", &bytes);
        let h = read_header(&path).unwrap();
        assert_eq!(h.head_count_kv_per_layer, Some(vec![2, 2, 2, 4]));
        assert_eq!(h.head_count_kv, None, "array form is not a scalar");
        assert_eq!(h.key_length, Some(512));
        assert_eq!(h.value_length_swa, Some(256));
        assert_eq!(h.swa_layer_flags, Some(vec![true, true, true, false]));
        std::fs::remove_file(path).ok();
    }

    /// DiffusionGemma's shape: the canvas key is literal (not arch-prefixed)
    /// and attention is non-causal.
    #[test]
    fn diffusion_header() {
        let bytes = synth_gguf(&[
            ("general.architecture", SynthVal::Str("diffusion-gemma")),
            ("diffusion-gemma.block_count", SynthVal::U32(30)),
            ("diffusion-gemma.attention.causal", SynthVal::Bool(false)),
            ("diffusion.canvas_length", SynthVal::U32(256)),
            ("tokenizer.ggml.tokens", SynthVal::StrArray(vec!["a", "bb", "ccc"])),
        ]);
        let path = write_temp("diffusion", &bytes);
        let h = read_header(&path).unwrap();
        assert_eq!(h.diffusion_canvas_length, Some(256));
        assert_eq!(h.attention_causal, Some(false));
        assert_eq!(h.vocab_size, Some(3));
        assert!(h.is_diffusion());
        assert!(h.runner_supported());
        assert_eq!(h.engine(), crate::profile::Engine::DiffusionGemma);
        std::fs::remove_file(path).ok();

        // Another diffusion architecture: still not llama-server's, but not
        // one the DiffusionGemma runner can load either.
        let bytes = synth_gguf(&[("general.architecture", SynthVal::Str("llada"))]);
        let path = write_temp("llada", &bytes);
        let h = read_header(&path).unwrap();
        assert!(h.is_diffusion());
        assert!(!h.runner_supported());
        assert_eq!(h.engine(), crate::profile::Engine::DiffusionGemma);
        std::fs::remove_file(path).ok();

        let bytes = synth_gguf(&[
            ("general.architecture", SynthVal::Str("llama")),
            ("llama.block_count", SynthVal::U32(32)),
            ("llama.attention.causal", SynthVal::Bool(true)),
            ("tokenizer.ggml.tokens", SynthVal::StrArray(vec!["a", "b"])),
        ]);
        let path = write_temp("llama", &bytes);
        let h = read_header(&path).unwrap();
        assert_eq!(h.diffusion_canvas_length, None);
        assert_eq!(h.attention_causal, Some(true));
        assert_eq!(h.vocab_size, Some(2));
        assert!(!h.is_diffusion());
        assert!(!h.runner_supported());
        assert_eq!(h.engine(), crate::profile::Engine::LlamaServer);
        std::fs::remove_file(path).ok();
    }

    /// The layout real converters write: general and architecture keys, the
    /// tokenizer's model and pre names, then the big tokenizer arrays, then
    /// keys llama-quantize and gguf-split append at the end.
    fn k2_like() -> Vec<u8> {
        let tokens: Vec<&'static str> = (0..300).map(|i| if i % 2 == 0 { "tok" } else { "en" }).collect();
        let merges: Vec<&'static str> = (0..200).map(|_| "t ok").collect();
        synth_gguf_with_tensors(
            &[
                ("general.architecture", SynthVal::Str("k2-horizon")),
                ("general.name", SynthVal::Str("Checkpoint_0002500")),
                ("k2-horizon.block_count", SynthVal::U32(36)),
                ("k2-horizon.context_length", SynthVal::U32(524288)),
                ("k2-horizon.attention.head_count", SynthVal::U32(32)),
                ("k2-horizon.attention.head_count_kv", SynthVal::U32(8)),
                ("k2-horizon.rope.scaling.type", SynthVal::Str("yarn")),
                // Gemma 4 writes a small tokenizer array before the model name.
                ("tokenizer.ggml.suppress_tokens", SynthVal::I32Array(vec![3, 7])),
                ("tokenizer.ggml.model", SynthVal::Str("gpt2")),
                ("tokenizer.ggml.pre", SynthVal::Str("k2-horizon")),
                ("tokenizer.ggml.tokens", SynthVal::StrArray(tokens)),
                ("tokenizer.ggml.token_type", SynthVal::I32Array(vec![1; 300])),
                ("tokenizer.ggml.merges", SynthVal::StrArray(merges)),
                ("tokenizer.ggml.eos_token_id", SynthVal::U32(1)),
                ("general.file_type", SynthVal::U32(15)),
                ("split.no", SynthVal::U16(0)),
                ("split.count", SynthVal::U16(3)),
            ],
            &[
                ("token_embd.weight", 12, &[4096, 250624]),
                ("blk.0.attn_q.weight", 12, &[4096, 4096]),
                ("blk.0.attn_norm.weight", 0, &[4096]),
                ("output.weight", 14, &[4096, 250624]),
            ],
        )
    }

    #[test]
    fn full_and_until_tokenizer_modes() {
        let bytes = k2_like();
        let label = Path::new("hf://IFM/K2@abc/k2.gguf");
        let size = bytes.len() as u64 * 1000; // the tensor data would follow

        let full = read_header_from(&mut std::io::Cursor::new(&bytes), size, label, ReadMode::Full).unwrap();
        assert_eq!(full.path, label);
        assert_eq!(full.file_size, size);
        assert!(!full.partial);
        assert_eq!(full.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(full.tokenizer_model.as_deref(), Some("gpt2"));
        assert_eq!(full.tokenizer_pre.as_deref(), Some("k2-horizon"));
        assert_eq!(full.rope_scaling_type.as_deref(), Some("yarn"));
        assert_eq!(full.vocab_size, Some(300));
        assert_eq!(full.file_type, Some(15));
        assert_eq!((full.split_no, full.split_count), (Some(0), Some(3)));
        assert_eq!(full.tensor_count, 4);
        assert_eq!(full.tensors.len(), 4);
        assert_eq!(
            full.tensors[1],
            TensorInfo { name: "blk.0.attn_q.weight".into(), ggml_type: 12, dims: vec![4096, 4096] }
        );
        assert_eq!(full.max_tensor_type_id, Some(14));
        // Shard 1 of 3: the other two list their own tensors.
        assert!(full.tensor_types_partial);
        assert_eq!(full.max_tensor_type(), None);
        let mut whole = full.clone();
        whole.fold_split_shards(vec![full.clone(), full.clone()]);
        assert_eq!(whole.max_tensor_type(), Some(14));

        // The table is summarised, not serialised; the summary survives.
        let json = serde_json::to_value(&whole).unwrap();
        assert!(json.get("tensors").is_none());
        assert!(json.get("partial").is_none(), "only a partial read says so");
        assert!(json.get("tensor_types_partial").is_none());
        let back: GgufHeader = serde_json::from_value(json).unwrap();
        assert!(back.tensors.is_empty());
        assert_eq!(back.max_tensor_type(), Some(14));
        let back: GgufHeader = serde_json::from_value(serde_json::to_value(&full).unwrap()).unwrap();
        assert_eq!(back.max_tensor_type(), None, "a lone shard stays unknown");

        let early =
            read_header_from(&mut std::io::Cursor::new(&bytes), size, label, ReadMode::UntilTokenizer).unwrap();
        assert!(early.partial);
        assert_eq!(early.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(early.block_count, Some(36));
        assert_eq!(early.head_count_kv, Some(8));
        assert_eq!(early.tokenizer_model.as_deref(), Some("gpt2"), "past the small suppress_tokens array");
        assert_eq!(early.tokenizer_pre.as_deref(), Some("k2-horizon"));
        assert_eq!(early.vocab_size, Some(300), "from the tokens array header alone");
        assert_eq!(
            early.metadata.get("tokenizer.ggml.suppress_tokens"),
            Some(&Value::Array(vec![Value::I64(3), Value::I64(7)]))
        );
        assert_eq!(early.file_type, None, "written after the tokenizer");
        assert_eq!(early.split_count, None);
        assert!(early.tensors.is_empty());
        assert_eq!(early.max_tensor_type(), None);
        assert_eq!(serde_json::to_value(&early).unwrap()["partial"], true);

        // The early stop needs only the bytes up to the tokens array header.
        let tokens_at = bytes.windows(21).position(|w| w == b"tokenizer.ggml.tokens").unwrap();
        let need = tokens_at + 21 + 4 + 4 + 8; // key, value type, elem type, length
        let h = read_header_from(&mut std::io::Cursor::new(&bytes[..need]), size, label, ReadMode::UntilTokenizer)
            .unwrap();
        assert_eq!(h.vocab_size, Some(300));
        match read_header_from(&mut std::io::Cursor::new(&bytes[..need - 1]), size, label, ReadMode::UntilTokenizer) {
            Err(Error::GgufTruncated { at, .. }) => assert_eq!(at, need as u64),
            other => panic!("expected truncation, got {other:?}"),
        }
    }

    /// Every prefix of a valid header either parses or reports truncation
    /// with an offset past what it was given (so fetching up to `at` makes
    /// progress) and within the file; never a panic or another error.
    #[test]
    fn every_cut_reports_truncation() {
        let bytes = k2_like();
        let label = Path::new("synthetic.gguf");
        for mode in [ReadMode::Full, ReadMode::UntilTokenizer] {
            let mut first_ok = None;
            for cut in 0..bytes.len() {
                let r = read_header_from(&mut std::io::Cursor::new(&bytes[..cut]), bytes.len() as u64, label, mode);
                match r {
                    Ok(_) => {
                        first_ok.get_or_insert(cut);
                    }
                    Err(Error::GgufTruncated { at, path }) => {
                        assert!(first_ok.is_none(), "{mode:?}: truncated at cut {cut} after parsing at a shorter one");
                        assert!(at > cut as u64 && at <= bytes.len() as u64, "{mode:?}: cut {cut} -> at {at}");
                        assert_eq!(path, label);
                    }
                    Err(e) => panic!("{mode:?}: cut {cut}: {e}"),
                }
            }
            match mode {
                ReadMode::Full => assert_eq!(first_ok, None, "a full parse needs every byte"),
                ReadMode::UntilTokenizer => assert!(first_ok.unwrap() < bytes.len() / 2),
            }
        }
        // Following `at` from an empty prefix converges on a parse.
        let mut have = 0usize;
        let mut rounds = 0;
        loop {
            rounds += 1;
            match read_header_from(&mut std::io::Cursor::new(&bytes[..have]), bytes.len() as u64, label, ReadMode::Full)
            {
                Ok(h) => {
                    assert_eq!(h.tensors.len(), 4);
                    break;
                }
                Err(Error::GgufTruncated { at, .. }) => have = at as usize,
                Err(e) => panic!("{e}"),
            }
            assert!(rounds < 10_000);
        }
    }

    /// A split set's shards each list their own tensors (the first can list
    /// none): folded together they give the model's highest type, unless
    /// a shard is missing or was read only up to its tokenizer.
    #[test]
    fn folding_split_shards() {
        let shard = |no: u16, tensors: &[(&str, u32, &[u64])]| {
            let bytes = synth_gguf_with_tensors(
                &[("general.architecture", SynthVal::Str("k2-horizon")), ("split.no", SynthVal::U16(no)), ("split.count", SynthVal::U16(3))],
                tensors,
            );
            let size = bytes.len() as u64 + 1000;
            read_header_from(&mut std::io::Cursor::new(bytes), size, Path::new("s"), ReadMode::Full).unwrap()
        };
        let first = shard(0, &[]);
        let second = shard(1, &[("blk.0.ffn_up.weight", 12, &[8, 8])]);
        let third = shard(2, &[("blk.1.ffn_up.weight", 101, &[8, 8]), ("output.weight", 14, &[8])]);
        assert!([&first, &second, &third].iter().all(|h| h.max_tensor_type().is_none()));

        let mut h = first.clone();
        h.fold_split_shards(vec![second.clone(), third.clone()]);
        assert_eq!(h.max_tensor_type(), Some(101));
        assert_eq!(h.tensor_count, 3);
        assert_eq!(h.tensors.len(), 3);
        assert_eq!(h.file_size, first.file_size + second.file_size + third.file_size);

        let mut short = first.clone();
        short.fold_split_shards(vec![second.clone()]);
        assert_eq!(short.max_tensor_type(), None, "a shard is missing");
        let mut early = third.clone();
        early.partial = true;
        let mut h = first;
        h.fold_split_shards(vec![second, early]);
        assert_eq!(h.max_tensor_type(), None, "a shard read only to its tokenizer");
    }

    /// A local file cut short is reported as truncated, not as an I/O error.
    #[test]
    fn truncated_file_on_disk() {
        let bytes = k2_like();
        let path = write_temp("cut", &bytes[..bytes.len() - 3]);
        match read_header(&path) {
            Err(Error::GgufTruncated { at, .. }) => assert_eq!(at, bytes.len() as u64),
            other => panic!("expected truncation, got {other:?}"),
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn implausible_tensor_table_is_malformed() {
        let mut bytes = synth_gguf_with_tensors(&[], &[("x", 0, &[1, 2])]);
        // n_dims follows the 24-byte file header, the 8-byte name length and
        // the 1-byte name.
        let at = 4 + 4 + 8 + 8 + 8 + 1;
        bytes[at..at + 4].copy_from_slice(&99u32.to_le_bytes());
        let r = read_header_from(&mut std::io::Cursor::new(&bytes), bytes.len() as u64, Path::new("x"), ReadMode::Full);
        assert!(matches!(r, Err(Error::GgufMalformed { .. })), "{r:?}");
    }

    /// The first 64 KiB of ngquocvinh/K2-Horizon-7B-GGUF
    /// K2-Horizon-7B-Q4_K_M.gguf, an HTTP Range read at commit 223e6f68
    /// (the file is 5,592,219,008 bytes).
    #[test]
    fn real_k2_horizon_prefix() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gguf/k2-horizon-7b-q4_k_m.head64k.bin");
        let bytes = std::fs::read(&path).unwrap();
        let size = 5_592_219_008;
        let label = Path::new("hf://ngquocvinh/K2-Horizon-7B-GGUF/K2-Horizon-7B-Q4_K_M.gguf");
        let h = read_header_from(&mut std::io::Cursor::new(&bytes), size, label, ReadMode::UntilTokenizer).unwrap();
        assert_eq!(h.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(h.tokenizer_model.as_deref(), Some("gpt2"));
        assert_eq!(h.tokenizer_pre.as_deref(), Some("k2-horizon"));
        assert_eq!(h.vocab_size, Some(250_624));
        assert_eq!(h.block_count, Some(36));
        assert_eq!(h.context_length, Some(524_288));
        assert_eq!(h.head_count, Some(32));
        assert_eq!(h.head_count_kv, Some(8));
        assert_eq!(h.file_size, size);
        assert!(h.partial);
        // The vocabulary alone is megabytes: a full parse needs more.
        match read_header_from(&mut std::io::Cursor::new(&bytes), size, label, ReadMode::Full) {
            Err(Error::GgufTruncated { at, .. }) => assert!(at > bytes.len() as u64),
            other => panic!("expected truncation, got {other:?}"),
        }
    }

    #[test]
    fn file_type_names_follow_llama_h() {
        assert_eq!(file_type_name(15), "Q4_K_M");
        assert_eq!(file_type_name(30), "IQ4_XS", "30 is IQ4_XS, not BF16");
        assert_eq!(file_type_name(32), "BF16");
        assert_eq!(file_type_name(21), "Q2_K_S");
        assert_eq!(file_type_name(38), "MXFP4_MOE");
        assert_eq!(file_type_name(40), "Q1_0");
        assert_eq!(file_type_name(5), "file_type 5", "retired");
        assert_eq!(file_type_name(103), "file_type 103");
        assert_eq!(ggml_type_name(30), "BF16");
        assert_eq!(ggml_type_name(12), "Q4_K");
        assert_eq!(ggml_type_name(101), "type 101");
    }

    #[test]
    fn split_names() {
        assert_eq!(
            split_name("gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf"),
            Some(("gemma-4-26B-A4B-it-BF16", 1, 2))
        );
        assert_eq!(
            split_name("K2-Horizon-375B-A23B-Q8_0-00030-of-00030.GGUF"),
            Some(("K2-Horizon-375B-A23B-Q8_0", 30, 30))
        );
        assert_eq!(split_name("model-00003-of-00002.gguf"), None, "past the count");
        assert_eq!(split_name("model-00000-of-00002.gguf"), None);
        assert_eq!(split_name("-00001-of-00002.gguf"), None, "no prefix");
        assert_eq!(split_name("model-0001-of-00002.gguf"), None);
        assert_eq!(split_name("model-00001-of-00002.gguf.part"), None);
        assert_eq!(split_name("model-Q4_K_M.gguf"), None);
        assert_eq!(split_name("\u{e9}-00001-of-00002.gguf"), Some(("\u{e9}", 1, 2)));
        assert_eq!(split_name("\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}.gguf"), None);
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
        // Point FIDIM_TEST_MODELS at a folder of GGUF files to run this.
        let Some(root_s) = std::env::var_os("FIDIM_TEST_MODELS") else { return };
        let root = std::path::Path::new(&root_s);
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
                    assert_eq!(h.tensors.len() as u64, h.tensor_count, "{p:?}");
                    // The early stop agrees with the full parse on what it reads.
                    let mut r = BufReader::new(File::open(&p).unwrap());
                    let early = read_header_from(&mut r, h.file_size, &p, ReadMode::UntilTokenizer).unwrap();
                    assert_eq!(early.architecture, h.architecture, "{p:?}");
                    assert_eq!(early.tokenizer_model, h.tokenizer_model, "{p:?}");
                    assert_eq!(early.tokenizer_pre, h.tokenizer_pre, "{p:?}");
                    assert_eq!(early.vocab_size, h.vocab_size, "{p:?}");
                    assert_eq!(early.block_count, h.block_count, "{p:?}");
                    eprintln!(
                        "{}: {} tensors, max type {:?}, pre {:?}",
                        p.display(),
                        h.tensor_count,
                        h.max_tensor_type().map(ggml_type_name),
                        h.tokenizer_pre
                    );
                    checked += 1;
                }
            }
        }
        eprintln!("parsed {checked} real GGUF files");
    }
}
