//! Process-lifetime plumbing for fidim-dg: the kill-on-close Job Object that
//! ties the runner to the helper, and the error mode that keeps a crashing
//! runner from parking behind a dialog box.
//!
//! The helper never duplicates the job handle, so any death of the helper
//! (taskkill, panic, crash) closes it and Windows kills the runner with it:
//! a ~16 GB VRAM allocation can never outlive its supervisor.

#[cfg(windows)]
mod imp {
    use std::os::windows::io::AsRawHandle;

    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::Diagnostics::Debug::{
        SetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX,
    };
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct KillOnCloseJob(HANDLE);

    // A job handle is a kernel object reference, usable from any thread.
    unsafe impl Send for KillOnCloseJob {}
    unsafe impl Sync for KillOnCloseJob {}

    impl KillOnCloseJob {
        pub fn new() -> std::io::Result<Self> {
            let h = unsafe { CreateJobObjectW(None, PCWSTR::null()) }?;
            // Owned from here on: an early return below still closes it.
            let job = KillOnCloseJob(h);
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const core::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            }?;
            Ok(job)
        }

        pub fn assign(&self, c: &std::process::Child) -> std::io::Result<()> {
            unsafe { AssignProcessToJobObject(self.0, HANDLE(c.as_raw_handle())) }?;
            Ok(())
        }
    }

    impl Drop for KillOnCloseJob {
        fn drop(&mut self) {
            let _ = unsafe { CloseHandle(self.0) };
        }
    }

    /// SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX for this process and,
    /// by inheritance, the runner: a missing DLL or an `abort()` then exits
    /// instead of raising a dialog that keeps a dead runner holding VRAM.
    /// Called ONLY from fidim-dg's `main`: the mode is process-wide and every
    /// later child inherits it, which FIDIM's GUI/CLI must not impose on
    /// llama-server.
    pub fn quiet_error_mode() {
        unsafe {
            SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
        }
    }
}

#[cfg(not(windows))]
mod imp {
    /// Without Job Objects the runner's lifetime rests on stdin EOF alone.
    pub struct KillOnCloseJob;

    impl KillOnCloseJob {
        pub fn new() -> std::io::Result<Self> {
            Ok(KillOnCloseJob)
        }
        pub fn assign(&self, _c: &std::process::Child) -> std::io::Result<()> {
            Ok(())
        }
    }

    pub fn quiet_error_mode() {}
}

pub use imp::{quiet_error_mode, KillOnCloseJob};
