//! A scripted HTTP/1.1 server on loopback for the tests of the network code
//! (Hub client, downloader). One connection at a time, `Connection: close`
//! on every response, every request logged for assertions.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Req {
    pub method: String,
    /// Path and query, as sent.
    pub target: String,
    /// Names lowercased.
    pub headers: Vec<(String, String)>,
}

impl Req {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str())
    }
    /// The path without the query string.
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or("")
    }
}

pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// Send only this many body bytes, then close, while Content-Length
    /// still announces the whole body: a connection dropped mid-transfer.
    pub cut_after: Option<usize>,
}

impl Resp {
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Resp { status, headers: Vec::new(), body: body.into(), cut_after: None }
    }
    pub fn header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.to_string(), value.into()));
        self
    }
    pub fn cut_after(mut self, n: usize) -> Self {
        self.cut_after = Some(n);
        self
    }
}

pub struct Server {
    /// `http://<ip>:<port>`
    pub base: String,
    log: Arc<Mutex<Vec<Req>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    /// Serve on `ip` (e.g. `127.0.0.1`) at a free port. Err when that
    /// address cannot be bound (a second loopback address on some systems).
    pub fn start(ip: &str, handler: impl Fn(&Req) -> Resp + Send + 'static) -> std::io::Result<Server> {
        let listener = TcpListener::bind((ip, 0))?;
        listener.set_nonblocking(true)?;
        let base = format!("http://{ip}:{}", listener.local_addr()?.port());
        let log = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (log2, stop2) = (log.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = serve(stream, &handler, &log2);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Server { base, log, stop, thread: Some(thread) })
    }

    pub fn requests(&self) -> Vec<Req> {
        self.log.lock().unwrap().clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Answer one request. It is logged before the response goes out, so a
/// client that has its answer can already see it in the log.
fn serve(mut stream: TcpStream, handler: &dyn Fn(&Req) -> Resp, log: &Mutex<Vec<Req>>) -> Option<()> {
    // Accepted sockets inherit the listener's non-blocking mode on Windows.
    stream.set_nonblocking(false).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => head.push(byte[0]),
            _ => return None,
        }
        if head.len() > 64 * 1024 {
            return None;
        }
    }
    let text = String::from_utf8_lossy(&head);
    let mut lines = text.split("\r\n");
    let mut first = lines.next()?.split(' ');
    let method = first.next()?.to_string();
    let target = first.next()?.to_string();
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let req = Req { method, target, headers };
    log.lock().unwrap().push(req.clone());
    let resp = handler(&req);
    let mut out = format!("HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n", resp.status, resp.body.len());
    for (k, v) in &resp.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    let body = match resp.cut_after {
        Some(n) => &resp.body[..n.min(resp.body.len())],
        None => &resp.body[..],
    };
    let _ = stream.write_all(out.as_bytes());
    if req.method != "HEAD" {
        let _ = stream.write_all(body);
    }
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Both);
    Some(())
}
