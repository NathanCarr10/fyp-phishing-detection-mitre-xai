param(
    [string]$PythonExe = $env:PYTHON_EXE
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $PythonExe) {
    $KnownPython = 'C:\Users\natha\AppData\Local\Python\pythoncore-3.14-64\python.exe'
    if (Test-Path $KnownPython) {
        $PythonExe = $KnownPython
    }
}

if (-not $PythonExe) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & py -3 -m streamlit run src\app.py
        exit $LASTEXITCODE
    }
}

if (-not $PythonExe) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $PythonExe = $PythonCommand.Source
    }
}

if (-not $PythonExe) {
    throw 'No Python interpreter was found. Set PYTHON_EXE or install the Python launcher.'
}

& $PythonExe -m streamlit run src\app.py
exit $LASTEXITCODE