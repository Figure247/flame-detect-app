$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$appDir = Join-Path $root 'dist\win-unpacked'
$backendSourceDir = Join-Path $root 'backend-dist\backend'
$backendTargetDir = Join-Path $appDir 'resources'
$iss = Join-Path $root 'installer\FlameDetectPro.iss'
$isccCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
    (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) { throw 'Inno Setup 6 ISCC.exe was not found.' }

Push-Location $root
try {
    & pyinstaller backend.spec --noconfirm --clean --distpath backend-dist --workpath build-backend
    if ($LASTEXITCODE -ne 0) { throw 'Backend build failed.' }

    Remove-Item -Path $appDir -Recurse -Force -ErrorAction SilentlyContinue
    & npx electron-builder --win dir --x64
    if ($LASTEXITCODE -ne 0) { throw 'Electron build failed.' }

    Copy-Item -Path (Join-Path $backendSourceDir '*') -Destination $backendTargetDir -Recurse -Force

    Remove-Item (Join-Path $appDir 'resources\main.py'), (Join-Path $appDir 'resources\start_backend.py') -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $appDir -Recurse -Filter 'pyvenv.cfg' -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Path $appDir -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('venv', '.venv', 'venv_backup', 'venv311') } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    & $iscc $iss
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup build failed.' }

    $output = Join-Path $root 'dist\FlameDetect-Pro-CUDA-Setup-2.0.1.exe'
    Get-Item $output | Select-Object FullName, Length, LastWriteTime
}
finally {
    Pop-Location
}
