#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif

#define MyAppName "YouTube Bili Localizer"
#define MyAppExeName "YouTubeBiliLocalizer.exe"

[Setup]
AppId={{C54E5924-3030-4D56-B130-5B28AA9A9D77}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=YouTube Bili Localizer contributors
DefaultDirName={localappdata}\Programs\YouTubeBiliLocalizer
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\releases
OutputBaseFilename=YouTubeBiliLocalizer-v{#MyAppVersion}-setup
SetupIconFile=..\assets\app-icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

#ifdef ChineseLanguageFile
[Languages]
Name: "chinesesimp"; MessagesFile: "{#ChineseLanguageFile}"
#endif

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: checkedonce

[Files]
Source: "..\dist\YouTubeBiliLocalizer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
