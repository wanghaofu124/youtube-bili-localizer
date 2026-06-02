Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = "D:\python\pythonw.exe"
command = """" & pythonw & """ -m yblocalizer.gui"
shell.Environment("PROCESS")("PYTHONPATH") = projectDir & "\src"
shell.CurrentDirectory = projectDir
shell.Run command, 0, False
