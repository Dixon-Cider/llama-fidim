#!/usr/bin/env bash
# Start the Llama FIDIM desktop app on Linux.
# Build: cd ui && npx tauri build --no-bundle -c '{"build":{"beforeBuildCommand":"npm run build"}}'
# From a terminal inside the desktop session the display variables are already set; from
# elsewhere (ssh, an agent) they are borrowed from any process of this user's graphical session.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/target/release/llama-fidim"
[ -x "$BIN" ] || { echo "not built: $BIN" >&2; exit 1; }
if [ -z "${WAYLAND_DISPLAY:-}${DISPLAY:-}" ]; then
  for p in /proc/[0-9]*; do
    [ "$(stat -c %u "$p" 2>/dev/null)" = "$(id -u)" ] || continue
    env_line=$(tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -E '^(DISPLAY|WAYLAND_DISPLAY|XDG_RUNTIME_DIR|DBUS_SESSION_BUS_ADDRESS|XAUTHORITY)=' || true)
    if echo "$env_line" | grep -qE '^(WAYLAND_DISPLAY|DISPLAY)='; then export $(echo "$env_line" | xargs); break; fi
  done
fi
# A desktop entry and icons for this build, so the taskbar shows the app's own
# icon instead of the generic one (KDE/GNOME match the window's app id to a
# .desktop file of that name). Idempotent; the .deb installs the same files.
ICONS="$HERE/ui/src-tauri/icons"
for name in llama-fidim rocks.fca.fidim; do
  for s in 32 128 256 512; do
    mkdir -p "$HOME/.local/share/icons/hicolor/${s}x${s}/apps"
    cp "$ICONS/${s}x${s}.png" "$HOME/.local/share/icons/hicolor/${s}x${s}/apps/$name.png"
  done
  mkdir -p "$HOME/.local/share/icons/hicolor/scalable/apps" "$HOME/.local/share/applications"
  cp "$ICONS/icon.svg" "$HOME/.local/share/icons/hicolor/scalable/apps/$name.svg"
  sed -e "s#^Exec=.*#Exec=$BIN#" -e "s#^Icon=.*#Icon=$name#" -e "s#^StartupWMClass=.*#StartupWMClass=$name#" \
    "$HERE/ui/src-tauri/linux/rocks.fca.fidim.desktop" > "$HOME/.local/share/applications/$name.desktop"
done
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
pkill -u "$USER" -x llama-fidim 2>/dev/null || true
mkdir -p "$HOME/.fidim"
nohup "$BIN" >> "$HOME/.fidim/ui-linux.log" 2>&1 &
echo "Llama FIDIM started (pid $!), log $HOME/.fidim/ui-linux.log"
