; Inno Setup script for the Windows installer, DicomDoodle-<version>-Setup.exe.
; The build workflow (.github/workflows/windows-installer.yml) compiles it after PyInstaller:
;     ISCC.exe /DAppVersion=2.0.0 DicomDoodle_App\packaging\DicomDoodle.iss
; It packs build\dist\DicomDoodle\ (made by DicomDoodle.spec) into build\installer\.

#ifndef AppVersion
  #define AppVersion "2.0.0"
#endif

[Setup]
; AppId identifies DicomDoodle to Windows: keep it, or new versions install side by side
AppId={{0DAAAA76-CAE7-423D-9BEA-1648E6108B85}
AppName=DicomDoodle
AppVersion={#AppVersion}
AppVerName=DicomDoodle {#AppVersion}
AppPublisher=Pushp Lochan Kumar
AppPublisherURL=https://github.com/pushplochan/DicomDoodle
AppSupportURL=https://github.com/pushplochan/DicomDoodle/issues
DefaultDirName={autopf}\DicomDoodle
DisableProgramGroupPage=yes
; No administrator rights needed (installs for the current user); the first page
; offers "all users" to those who have them
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
; file.ico's picture as a real Windows icon (file.ico is a PNG, which Inno Setup refuses)
SetupIconFile=DicomDoodle.ico
UninstallDisplayIcon={app}\DicomDoodle.exe
OutputDir=..\..\build\installer
OutputBaseFilename=DicomDoodle-{#AppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\build\dist\DicomDoodle\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; files of the previous version (an update installs over it)
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\DicomDoodle"; Filename: "{app}\DicomDoodle.exe"
Name: "{autodesktop}\DicomDoodle"; Filename: "{app}\DicomDoodle.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\DicomDoodle.exe"; Description: "{cm:LaunchProgram,DicomDoodle}"; Flags: nowait postinstall skipifsilent
