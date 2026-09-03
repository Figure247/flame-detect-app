Option Explicit

Dim WshShell, scriptDir, batchPath
Set WshShell = CreateObject("WScript.Shell")

scriptDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
batchPath = scriptDir & "\start_app.bat"
WshShell.CurrentDirectory = scriptDir
WshShell.Run "cmd.exe /d /c call " & Chr(34) & batchPath & Chr(34), 0, False

Set WshShell = Nothing
