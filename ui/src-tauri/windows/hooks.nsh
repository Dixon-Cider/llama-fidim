; Hooks for Llama FIDIM's NSIS installer, named by tauri.release.conf.json
; (bundle > windows > nsis > installerHooks). Tauri's installer template
; includes this file after its own headers (LogicLib, FileFunc and its
; utils.nsh) and inserts the NSIS_HOOK_* macros into its sections:
;
;   Install    SetOutPath $INSTDIR, PREINSTALL, close a running GUI, copy the
;              files, write the uninstaller, registry and shortcuts, POSTINSTALL
;   Uninstall  PREUNINSTALL, close a running GUI, delete the files
;
; Closing the GUI can stop the install: the user answers Cancel to Tauri's
; prompt, or the GUI cannot be killed (it runs elevated). So PREINSTALL and
; PREUNINSTALL run that same check first, before they change anything, and
; Tauri's own check right after them finds nothing left to close.
;
; Tauri closes only llama-fidim.exe (it terminates every copy the user runs).
; fidim.exe and fidim-dg.exe can be running as well, detached from the GUI:
; a DiffusionGemma server (fidim-dg.exe) holding a model in VRAM, or a
; keep-alive helper (fidim.exe). They must outlive an install, an upgrade
; and an uninstall, but Windows refuses to overwrite or delete a running
; executable. It does let one be renamed: the process keeps running from
; the renamed file with the same pid and image name, so `fidim stop` still
; finds it, and a later install deletes the renamed copy once it has
; exited. scripts/install.ps1 follows the same rule.
;
; An interactive upgrade (the reinstall page's default choice) runs the
; installed version's uninstaller first, with that version's hooks, so
; PREUNINSTALL moves running helpers aside as well. A silent or passive
; upgrade installs over the old version without it.
;
; Macro arguments are pasted into the macro body: never pass a register
; ($0-$9, $R0-$R9) as one, the macros use them.

!include LogicLib.nsh
!include FileFunc.nsh

; Where scripts/install.ps1 installed Llama FIDIM before it moved to the
; installer's folder. Its Start Menu entry had the name the installer uses.
; (scripts/test-installer-hooks.ps1 defines both first, pointing into a
; temporary folder.)
!define /ifndef FIDIM_OLD_INSTALL_DIR "$LOCALAPPDATA\Programs\LlamaFIDIM"
!define /ifndef FIDIM_OLD_SHORTCUT "$SMPROGRAMS\Llama FIDIM.lnk"

; Delete ${DIR}\${NAME}, or rename it to ${NAME}.old-<yyyyMMddHHmmss> when it
; is running. First deletes the renamed copies earlier runs left there,
; except those still running.
!macro FIDIM_MOVE_ASIDE DIR NAME
  Push $R0
  Push $R1
  Push $R2
  Push $R3
  Push $R4
  Push $R5
  Push $R6
  Push $R7

  ClearErrors
  FindFirst $R0 $R1 "${DIR}\${NAME}.old-*"
  ${DoWhile} $R1 != ""
    Delete "${DIR}\$R1"
    FindNext $R0 $R1
  ${Loop}
  FindClose $R0

  ${If} ${FileExists} "${DIR}\${NAME}"
    ClearErrors
    Delete "${DIR}\${NAME}"
    ${If} ${Errors}
      ; day, month, year, weekday, hour, minute, second; local time, zero padded
      ${GetTime} "" "L" $R1 $R2 $R3 $R4 $R5 $R6 $R7
      StrCpy $R0 "${NAME}.old-$R3$R2$R1$R5$R6$R7"
      ; A second move within the same second (an upgrade's old uninstaller,
      ; then the installer) must not collide with the first.
      StrCpy $R1 $R0
      StrCpy $R2 1
      ${DoWhile} ${FileExists} "${DIR}\$R1"
        IntOp $R2 $R2 + 1
        StrCpy $R1 "$R0-$R2"
      ${Loop}
      ClearErrors
      Rename "${DIR}\${NAME}" "${DIR}\$R1"
      ${If} ${Errors}
        DetailPrint "Could not move the running ${DIR}\${NAME} aside."
      ${Else}
        DetailPrint "Moved the running ${NAME} aside as $R1; it keeps running."
      ${EndIf}
    ${EndIf}
  ${EndIf}
  ClearErrors

  Pop $R7
  Pop $R6
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Pop $R1
  Pop $R0
!macroend

; Set $R1 to 1 when ${DIR} and ${OTHER} are the same folder, however they
; are spelled: with `.\` in the path, as a short 8.3 name, through a
; junction. Comparing the paths as text misses those, and the old folder's
; retirement would then delete the files just installed into it. A file
; created in ${DIR} shows up in ${OTHER} only when they are one folder (it
; is deleted again). When no file can be created in ${DIR}, $R1 is 1 as
; well, so the callers leave ${OTHER} alone. Uses $R0 and $R1; the caller
; saves them.
!macro FIDIM_SAME_FOLDER DIR OTHER
  StrCpy $R1 0
  ${If} ${FileExists} "${OTHER}\*.*"
    ClearErrors
    GetTempFileName $R0 "${DIR}"
    ${If} ${Errors}
    ${OrIf} $R0 == ""
      StrCpy $R1 1
    ${Else}
      ${GetFileName} $R0 $R1
      ${If} ${FileExists} "${OTHER}\$R1"
        StrCpy $R1 1
      ${Else}
        StrCpy $R1 0
      ${EndIf}
      Delete $R0
    ${EndIf}
    ClearErrors
  ${EndIf}
!macroend

; Delete the shortcut ${LNK} (and its Start and taskbar pins) when it starts
; ${OLDDIR}\llama-fidim.exe, the entry scripts/install.ps1 created, unless
; ${OLDDIR} is ${NEWDIR}, the folder being installed into. The installer
; writes its own entry under that name right after PREINSTALL.
; IsShortcutTarget and UnpinShortcut come from Tauri's utils.nsh.
!macro FIDIM_RETIRE_OLD_SHORTCUT LNK OLDDIR NEWDIR
  ${If} ${FileExists} "${LNK}"
    Push $0
    Push $1
    Push $2
    Push $3
    Push $R0
    Push $R1
    !insertmacro FIDIM_SAME_FOLDER "${NEWDIR}" "${OLDDIR}"
    ${If} $R1 = 0
      !insertmacro IsShortcutTarget "${LNK}" "${OLDDIR}\llama-fidim.exe"
      Pop $0
      ${If} $0 = 1
        !insertmacro UnpinShortcut "${LNK}"
        Delete "${LNK}"
        DetailPrint "Removed the old Start Menu entry for ${OLDDIR}."
      ${EndIf}
    ${EndIf}
    Pop $R1
    Pop $R0
    Pop $3
    Pop $2
    Pop $1
    Pop $0
  ${EndIf}
!macroend

; Retire what scripts/install.ps1 put in ${OLDDIR}: its three executables,
; with running ones moved aside, and then the folder if nothing else is in
; it; unless ${OLDDIR} is ${NEWDIR}, the folder just installed into. Runs
; after the GUI was closed, so only helpers can be running.
!macro FIDIM_RETIRE_OLD_INSTALL OLDDIR NEWDIR
  ${If} ${FileExists} "${OLDDIR}\*.*"
    Push $R0
    Push $R1
    !insertmacro FIDIM_SAME_FOLDER "${NEWDIR}" "${OLDDIR}"
    ${If} $R1 = 0
      !insertmacro FIDIM_MOVE_ASIDE "${OLDDIR}" "fidim-dg.exe"
      !insertmacro FIDIM_MOVE_ASIDE "${OLDDIR}" "fidim.exe"
      !insertmacro FIDIM_MOVE_ASIDE "${OLDDIR}" "llama-fidim.exe"
      ClearErrors
      RMDir "${OLDDIR}"
      ${IfNot} ${Errors}
        DetailPrint "Removed the old install folder ${OLDDIR}."
      ${EndIf}
      ClearErrors
    ${EndIf}
    Pop $R1
    Pop $R0
  ${EndIf}
!macroend

; The check Tauri runs right after PREINSTALL and PREUNINSTALL (its
; utils.nsh): find the user's running GUI, ask to kill it (a silent or
; passive run just kills it), and abort on Cancel or when it cannot be
; killed. Its labels are numbered by the line the hook is inserted at, so
; this copy does not collide with Tauri's in the same section.
!macro FIDIM_CLOSE_APP
  !insertmacro CheckIfAppIsRunning "${MAINBINARYNAME}.exe" "${PRODUCTNAME}"
!macroend

; On a first install Tauri's section creates $INSTDIR (SetOutPath) before
; PREINSTALL, so an install stopped at the app check, or failed any other
; way, would leave an empty folder behind. Remove it: RMDir without /r
; removes a folder only when it is empty. SetOutPath also made it the
; current directory, which Windows does not remove, so move out first.
; (Tauri's template has no .onInstFailed; should a later one bring its
; own, makensis stops at the duplicate.)
Function .onInstFailed
  SetOutPath $TEMP
  RMDir "$INSTDIR"
FunctionEnd

!macro NSIS_HOOK_PREINSTALL
  !insertmacro FIDIM_CLOSE_APP
  !insertmacro FIDIM_MOVE_ASIDE "$INSTDIR" "fidim-dg.exe"
  !insertmacro FIDIM_MOVE_ASIDE "$INSTDIR" "fidim.exe"
  !insertmacro FIDIM_RETIRE_OLD_SHORTCUT "${FIDIM_OLD_SHORTCUT}" "${FIDIM_OLD_INSTALL_DIR}" "$INSTDIR"
!macroend

!macro NSIS_HOOK_POSTINSTALL
  !insertmacro FIDIM_RETIRE_OLD_INSTALL "${FIDIM_OLD_INSTALL_DIR}" "$INSTDIR"
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro FIDIM_CLOSE_APP
  !insertmacro FIDIM_MOVE_ASIDE "$INSTDIR" "fidim-dg.exe"
  !insertmacro FIDIM_MOVE_ASIDE "$INSTDIR" "fidim.exe"
!macroend
