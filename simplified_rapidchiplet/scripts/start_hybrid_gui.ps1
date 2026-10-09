param([int]$Port = 8765, [string]$Out = 'simplified_rapidchiplet\results\hybrid_gui_cost16')
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = Join-Path $projectRoot '.venv/bin/python'
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python venv unavailable. Follow the installation steps in README.md first.'
}
$guiPath = Join-Path $projectRoot 'simplified_rapidchiplet\hybrid_gui.py'
$outputPath = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $projectRoot $Out }
& $pythonPath -X utf8 $guiPath --port $Port --out $outputPath
