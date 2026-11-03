' Runs backup-trigger.ps1 with no console window flashing up (used by Task Scheduler).
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & _
    CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & _
    "\backup-trigger.ps1""", 0, False
