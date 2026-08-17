$env:PYTHONPATH = "$PSScriptRoot\..\..\work\ml_deps"
& "C:\Users\Manis\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "$PSScriptRoot\dashboard_server.py" --data "$PSScriptRoot\artifacts\master_workforce_dataset.csv" --artifacts "$PSScriptRoot\artifacts" --port 8501
