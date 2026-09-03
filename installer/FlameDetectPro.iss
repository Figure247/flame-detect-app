#define AppVersion "2.0.1"
#define AppName "FlameDetect Pro"
#define AppPublisher "FlameDetect"
#define AppDir "..\dist\win-unpacked"
#define OutputDir "..\dist"

[Setup]
AppId={{B8A9E9B1-9A4E-4F18-9D66-4D6BE0CBBF17}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\FlameDetect Pro
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename=FlameDetect-Pro-CUDA-Setup-{#AppVersion}
SetupIconFile=..\assets\icon.ico
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
CloseApplications=yes
RestartApplications=yes
CloseApplicationsFilter=FlameDetect Pro.exe,backend.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\FlameDetect Pro.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#AppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\FlameDetect Pro.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\FlameDetect Pro.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FlameDetect Pro.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
