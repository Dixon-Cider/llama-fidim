//! The two byte-level decoders under the chat client, both incremental so a
//! reply can be shown while it is still arriving:
//!
//! - `ChunkedReader`: `Transfer-Encoding: chunked` as a `Read`. The existing
//!   `supervise::dechunk` needs the whole body; a stream never has one.
//!   It keeps its own buffer, so a `WouldBlock` from a non-blocking socket
//!   underneath can surface at any byte and the next `read` resumes there.
//! - `SseParser`: server-sent events. `data:` lines (a multi-line event is
//!   joined with `\n`) and `:` comments, which fidim-dg uses for queue and
//!   denoise progress and llama-server for keep-alive pings.

use std::io::{self, Read};

/// Longest chunk-size or trailer line accepted; a real one is a few bytes.
const MAX_CHUNK_LINE: usize = 4096;
/// Longest SSE line kept. A final chunk with timings is ~1 KB; tool-call
/// arguments can be larger. Anything past this is dropped, not buffered.
const MAX_SSE_LINE: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------- chunked ----

#[derive(Debug, Clone, Copy, PartialEq)]
enum ChunkState {
    /// Expecting a `<hex>[;ext]` line.
    Size,
    /// This many data bytes left in the current chunk.
    Data(usize),
    /// The CRLF after a chunk's data.
    DataEnd,
    /// After the zero chunk: trailer lines until an empty one.
    Trailer,
    Done,
}

/// Decodes a chunked body from `inner` as it arrives.
pub struct ChunkedReader<R> {
    inner: R,
    /// Raw bytes from `inner` not yet decoded, from `pos`.
    buf: Vec<u8>,
    pos: usize,
    state: ChunkState,
}

impl<R: Read> ChunkedReader<R> {
    pub fn new(inner: R) -> Self {
        Self::with_prefix(inner, Vec::new())
    }

    /// `prefix` = body bytes that were read together with the headers.
    pub fn with_prefix(inner: R, prefix: Vec<u8>) -> Self {
        ChunkedReader { inner, buf: prefix, pos: 0, state: ChunkState::Size }
    }

    /// True once the zero chunk and its trailers have been read.
    pub fn is_done(&self) -> bool {
        self.state == ChunkState::Done
    }

    /// Read more raw bytes. EOF before the zero chunk is an error: the
    /// server went away mid-reply.
    fn fill(&mut self) -> io::Result<()> {
        if self.pos > 0 && self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        } else if self.pos > 64 * 1024 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        let mut tmp = [0u8; 16 * 1024];
        match self.inner.read(&mut tmp)? {
            0 => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the chunked body ended before its last chunk")),
            n => {
                self.buf.extend_from_slice(&tmp[..n]);
                Ok(())
            }
        }
    }

    /// The next line from the buffer without its line ending, or None when
    /// it has not fully arrived yet.
    fn take_line(&mut self) -> io::Result<Option<String>> {
        let rest = &self.buf[self.pos..];
        match rest.iter().position(|&b| b == b'\n') {
            Some(i) => {
                let mut line = &rest[..i];
                if line.last() == Some(&b'\r') {
                    line = &line[..line.len() - 1];
                }
                let s = String::from_utf8_lossy(line).into_owned();
                self.pos += i + 1;
                Ok(Some(s))
            }
            None if rest.len() > MAX_CHUNK_LINE => {
                Err(io::Error::new(io::ErrorKind::InvalidData, "a chunk-size line is too long"))
            }
            None => Ok(None),
        }
    }
}

impl<R: Read> Read for ChunkedReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            match self.state {
                ChunkState::Done => return Ok(0),
                ChunkState::Size => match self.take_line()? {
                    Some(line) => {
                        let hex = line.split(';').next().unwrap_or("").trim();
                        let size = usize::from_str_radix(hex, 16).map_err(|_| {
                            io::Error::new(io::ErrorKind::InvalidData, format!("bad chunk size {hex:?}"))
                        })?;
                        self.state = if size == 0 { ChunkState::Trailer } else { ChunkState::Data(size) };
                    }
                    None => self.fill()?,
                },
                ChunkState::Data(left) => {
                    let avail = self.buf.len() - self.pos;
                    if avail == 0 {
                        self.fill()?;
                        continue;
                    }
                    let k = left.min(avail).min(out.len());
                    out[..k].copy_from_slice(&self.buf[self.pos..self.pos + k]);
                    self.pos += k;
                    self.state = if left == k { ChunkState::DataEnd } else { ChunkState::Data(left - k) };
                    return Ok(k);
                }
                ChunkState::DataEnd => match self.take_line()? {
                    Some(l) if l.is_empty() => self.state = ChunkState::Size,
                    Some(_) => {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "chunk data is not followed by CRLF"))
                    }
                    None => self.fill()?,
                },
                ChunkState::Trailer => match self.take_line()? {
                    Some(l) if l.is_empty() => self.state = ChunkState::Done,
                    Some(_) => {}
                    None => self.fill()?,
                },
            }
        }
    }
}

/// Bytes already read (with the headers) first, then `inner`.
pub struct Prefixed<R> {
    head: Vec<u8>,
    pos: usize,
    inner: R,
}

impl<R> Prefixed<R> {
    pub fn new(head: Vec<u8>, inner: R) -> Self {
        Prefixed { head, pos: 0, inner }
    }
}

impl<R: Read> Read for Prefixed<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos < self.head.len() {
            let k = (self.head.len() - self.pos).min(out.len());
            out[..k].copy_from_slice(&self.head[self.pos..self.pos + k]);
            self.pos += k;
            return Ok(k);
        }
        self.inner.read(out)
    }
}

// -------------------------------------------------------------------- SSE ----

/// One thing an SSE stream delivered.
#[derive(Debug, Clone, PartialEq)]
pub enum SseItem {
    /// An event's data (multi-line data joined with `\n`).
    Data(String),
    /// A `:` comment line, without the colon and one leading space.
    Comment(String),
}

/// Incremental SSE parser (the WHATWG event-stream grammar, minus the
/// fields a chat has no use for: `event`, `id` and `retry` are ignored).
/// Line endings may be CRLF, LF or CR, split across pushes anywhere.
/// Newline bytes never occur inside a UTF-8 sequence, so lines are decoded
/// only once complete and a character split between two pushes survives.
#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    /// The line outgrew MAX_SSE_LINE: drop the rest of it.
    overflow: bool,
    data: Option<String>,
    /// The last byte was CR: a following LF belongs to the same line end.
    after_cr: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<SseItem>) {
        for &b in bytes {
            if self.after_cr {
                self.after_cr = false;
                if b == b'\n' {
                    continue;
                }
            }
            match b {
                b'\r' => {
                    self.after_cr = true;
                    self.end_line(out);
                }
                b'\n' => self.end_line(out),
                _ if self.overflow => {}
                _ if self.line.len() >= MAX_SSE_LINE => {
                    self.overflow = true;
                    self.line.clear();
                }
                _ => self.line.push(b),
            }
        }
    }

    /// The stream ended. An event still waiting for its blank line is
    /// delivered anyway: a server that closes without the final newline
    /// has still said what it said.
    pub fn finish(&mut self, out: &mut Vec<SseItem>) {
        if !self.line.is_empty() {
            self.end_line(out);
        }
        if let Some(d) = self.data.take() {
            out.push(SseItem::Data(d));
        }
    }

    fn end_line(&mut self, out: &mut Vec<SseItem>) {
        if std::mem::take(&mut self.overflow) {
            // A line too long to keep: neither data nor a dispatch.
            self.line.clear();
            return;
        }
        if self.line.is_empty() {
            if let Some(d) = self.data.take() {
                out.push(SseItem::Data(d));
            }
            return;
        }
        let line = std::mem::take(&mut self.line);
        let text = String::from_utf8_lossy(&line);
        if let Some(c) = text.strip_prefix(':') {
            out.push(SseItem::Comment(c.strip_prefix(' ').unwrap_or(c).to_string()));
            return;
        }
        let (field, value) = match text.find(':') {
            Some(i) => (&text[..i], text[i + 1..].strip_prefix(' ').unwrap_or(&text[i + 1..])),
            None => (&text[..], ""),
        };
        if field == "data" {
            match &mut self.data {
                Some(d) => {
                    d.push('\n');
                    d.push_str(value);
                }
                None => self.data = Some(value.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that hands out `data` in the given piece sizes (cycling),
    /// optionally returning WouldBlock between pieces, like a non-blocking
    /// socket whose bytes trickle in.
    struct Trickle {
        data: Vec<u8>,
        pos: usize,
        sizes: Vec<usize>,
        i: usize,
        would_block: bool,
        blocked: bool,
    }

    impl Trickle {
        fn new(data: &[u8], sizes: &[usize], would_block: bool) -> Self {
            Trickle { data: data.to_vec(), pos: 0, sizes: sizes.to_vec(), i: 0, would_block, blocked: false }
        }
    }

    impl Read for Trickle {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.would_block && !self.blocked {
                self.blocked = true;
                return Err(io::ErrorKind::WouldBlock.into());
            }
            self.blocked = false;
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let size = self.sizes[self.i % self.sizes.len()].max(1);
            self.i += 1;
            let k = size.min(out.len()).min(self.data.len() - self.pos);
            out[..k].copy_from_slice(&self.data[self.pos..self.pos + k]);
            self.pos += k;
            Ok(k)
        }
    }

    /// Drain a reader, retrying WouldBlock, reading `step` bytes at a time.
    fn drain(r: &mut impl Read, step: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut buf = vec![0u8; step];
        loop {
            match r.read(&mut buf) {
                Ok(0) => return Ok(out),
                Ok(n) => out.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                Err(e) => return Err(e),
            }
        }
    }

    fn chunked(parts: &[&str]) -> Vec<u8> {
        let mut v = Vec::new();
        for p in parts {
            v.extend_from_slice(format!("{:x}\r\n{p}\r\n", p.len()).as_bytes());
        }
        v.extend_from_slice(b"0\r\n\r\n");
        v
    }

    #[test]
    fn chunked_decodes_at_every_split() {
        let parts = ["data: {\"a\":1}\n\n", ": queued 1\n\n", "data: [DONE]\n\n", "é漢字🙂"];
        let raw = chunked(&parts);
        let want: String = parts.concat();
        // Every piece size from 1 byte up, with and without WouldBlock
        // between pieces, and every output buffer size.
        for piece in 1..=raw.len() {
            for wb in [false, true] {
                for step in [1, 3, 7, 4096] {
                    let mut r = ChunkedReader::new(Trickle::new(&raw, &[piece], wb));
                    let got = drain(&mut r, step).unwrap();
                    assert_eq!(String::from_utf8(got).unwrap(), want, "piece={piece} wb={wb} step={step}");
                    assert!(r.is_done());
                }
            }
        }
    }

    #[test]
    fn chunked_irregular_splits_extensions_and_trailers() {
        let raw = b"5;name=x\r\nhello\r\n1\r\n \r\n6\r\nworld!\r\n0\r\nX-Trailer: 1\r\n\r\n";
        for sizes in [&[1usize, 2, 3][..], &[7, 1], &[2, 11, 5], &[64]] {
            let mut r = ChunkedReader::new(Trickle::new(raw, sizes, true));
            assert_eq!(drain(&mut r, 5).unwrap(), b"hello world!", "{sizes:?}");
        }
        // Bytes read together with the headers come first.
        let (head, tail) = raw.split_at(9);
        let mut r = ChunkedReader::with_prefix(Trickle::new(tail, &[4], false), head.to_vec());
        assert_eq!(drain(&mut r, 64).unwrap(), b"hello world!");
        // Bare LF line endings are tolerated.
        let mut r = ChunkedReader::new(&b"3\nabc\n0\n\n"[..]);
        assert_eq!(drain(&mut r, 64).unwrap(), b"abc");
    }

    #[test]
    fn chunked_errors() {
        // EOF mid-chunk: an error, not a silent short body.
        let mut r = ChunkedReader::new(&b"a\r\nhello"[..]);
        let e = drain(&mut r, 64).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        // A hex size that is not hex.
        let mut r = ChunkedReader::new(&b"zz\r\nhello\r\n"[..]);
        assert_eq!(drain(&mut r, 64).unwrap_err().kind(), io::ErrorKind::InvalidData);
        // Data not followed by CRLF.
        let mut r = ChunkedReader::new(&b"2\r\nabXY0\r\n\r\n"[..]);
        assert_eq!(drain(&mut r, 64).unwrap_err().kind(), io::ErrorKind::InvalidData);
        // An endless size line is refused rather than buffered.
        let long = vec![b'1'; MAX_CHUNK_LINE + 10];
        let mut r = ChunkedReader::new(&long[..]);
        assert_eq!(drain(&mut r, 64).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    fn parse_all(bytes: &[u8], piece: usize) -> Vec<SseItem> {
        let mut p = SseParser::new();
        let mut out = Vec::new();
        for c in bytes.chunks(piece.max(1)) {
            p.push(c, &mut out);
        }
        p.finish(&mut out);
        out
    }

    #[test]
    fn sse_data_comments_and_multiline() {
        let text = "data: {\"x\":1}\n\n: queued 2\n\n:\n\ndata:no-space\n\ndata: a\ndata: b\n\nevent: ping\nid: 7\nretry: 10\n\ndata: [DONE]\n\n";
        let want = vec![
            SseItem::Data("{\"x\":1}".into()),
            SseItem::Comment("queued 2".into()),
            SseItem::Comment("".into()),
            SseItem::Data("no-space".into()),
            SseItem::Data("a\nb".into()),
            SseItem::Data("[DONE]".into()),
        ];
        for piece in 1..=text.len() {
            assert_eq!(parse_all(text.as_bytes(), piece), want, "piece={piece}");
        }
        // CRLF and CR line endings, including a CRLF split across pushes.
        let crlf = text.replace('\n', "\r\n");
        let cr = text.replace('\n', "\r");
        for piece in 1..=crlf.len() {
            assert_eq!(parse_all(crlf.as_bytes(), piece), want, "crlf piece={piece}");
        }
        for piece in [1, 2, 5, 1000] {
            assert_eq!(parse_all(cr.as_bytes(), piece), want, "cr piece={piece}");
        }
    }

    #[test]
    fn sse_utf8_split_inside_a_character() {
        let text = "data: {\"content\":\"héllo 漢字 🙂\"}\n\n";
        for piece in 1..=text.len() {
            assert_eq!(parse_all(text.as_bytes(), piece), vec![SseItem::Data("{\"content\":\"héllo 漢字 🙂\"}".into())]);
        }
    }

    #[test]
    fn sse_unterminated_event_is_flushed_at_the_end() {
        assert_eq!(parse_all(b"data: [DONE]", 3), vec![SseItem::Data("[DONE]".into())]);
        assert_eq!(parse_all(b"data: x\n", 1), vec![SseItem::Data("x".into())]);
        assert!(parse_all(b"", 1).is_empty());
    }

    #[test]
    fn sse_overlong_lines_are_dropped() {
        let mut p = SseParser::new();
        let mut out = Vec::new();
        p.push(b"data: ", &mut out);
        let big = vec![b'x'; MAX_SSE_LINE + 5];
        p.push(&big, &mut out);
        p.push(b"\n\ndata: ok\n\n", &mut out);
        assert_eq!(out, vec![SseItem::Data("ok".into())]);
    }
}
