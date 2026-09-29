; Inno Setup script for Afono Nullpoint. Build dist\Afono Nullpoint first (python packaging/build.py),
; then compile this file with Inno Setup 6 to get dist\Afono-Nullpoint-Setup.exe.
#define AppVersion "0.1.0"

[Setup]
AppId={{6F1F0B7E-3C1A-4B7B-9E1D-0A0F00000001}
AppName=Afono Nullpoint
AppVersion={#AppVersion}
AppPublisher=Afono
AppPublisherURL=https://github.com/aamodpaudel/sEMG_research
DefaultDirName={autopf}\Afono Nullpoint
DefaultGroupName=Afono Nullpoint
OutputDir=..\dist
OutputBaseFilename=Afono-Nullpoint-Setup-{#AppVersion}
SetupIconFile=..\app\assets\afono.ico
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\Afono Nullpoint\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Afono Nullpoint"; Filename: "{app}\Afono Nullpoint.exe"
Name: "{autodesktop}\Afono Nullpoint"; Filename: "{app}\Afono Nullpoint.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\Afono Nullpoint.exe"; Description: "Start Afono Nullpoint"; Flags: nowait postinstall skipifsilent
