# Toolchain captures

- `vswhere.json`: `vswhere -all -products * -requires
  Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json` on a
  machine with Build Tools 2026, Build Tools 2022 and Community 2022,
  trimmed to a few fields and reordered so the newest is not first.
- `hip-cmath-clash.txt`: HIP SDK 7.1 clang compiling a `<cmath>`-using
  kernel for gfx1201 against MSVC 14.51, with an unpatched copy of the
  SDK's `__clang_hip_runtime_wrapper.h` (llama.cpp#22570). The copy's
  temporary path is replaced by the SDK's own.
