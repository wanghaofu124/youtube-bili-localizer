Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)

venvPythonw = projectDir & "\.venv\Scripts\pythonw.exe"
If fso.FileExists(venvPythonw) Then
    pythonw = venvPythonw
Else
    pythonw = "pythonw"
End If

command = """" & pythonw & """ -m yblocalizer.gui"
shell.Environment("PROCESS")("PYTHONPATH") = projectDir & "\src"
shell.CurrentDirectory = projectDir
shell.Run command, 0, False
