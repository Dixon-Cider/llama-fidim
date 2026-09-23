"""In-process RSS guard, active in every Python process that sees this on PYTHONPATH
(including SGLang's spawned scheduler/detokenizer). When a process crosses
SGL_MEMGUARD_GB (default 6), it writes every thread's stack to
SGL_MEMGUARD_LOG and exits, before the kernel OOM killer takes the desktop down.
Also raises oom_score_adj so that if something else blows up, we die first."""
import os, sys, threading, time, traceback

_LIMIT = float(os.environ.get("SGL_MEMGUARD_GB", "6"))
_LOG = os.environ.get("SGL_MEMGUARD_LOG", "/tmp/memguard.log")

def _rss_gb():
    """Anonymous RSS only: mmap'd weight files are file-backed and reclaimable,
    so they must not count (the OOM that hit this box was 21 GB of anon)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return 0.0

def _guard():
    peak = 0.0
    while True:
        r = _rss_gb()
        peak = max(peak, r)
        if r > _LIMIT:
            with open(_LOG, "a") as f:
                f.write(f"\n===== MEMGUARD pid={os.getpid()} rss={r:.1f} GB argv={' '.join(sys.argv)[:200]} =====\n")
                for tid, frame in sys._current_frames().items():
                    f.write(f"--- thread {tid} ---\n")
                    f.write("".join(traceback.format_stack(frame)))
                f.flush()
            os._exit(3)
        time.sleep(0.25)

try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
except Exception:
    pass
threading.Thread(target=_guard, daemon=True, name="memguard").start()
