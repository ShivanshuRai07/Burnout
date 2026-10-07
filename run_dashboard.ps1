$pythonExe = "python"
$codexPy = "C:\Users\Manis\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} elseif (Test-Path $codexPy) {
    $pythonExe = $codexPy
}

& $pythonExe "$PSScriptRoot\dashboard_server.py" --artifacts "$PSScriptRoot\artifacts" --port 8501

