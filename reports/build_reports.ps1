param(
    [ValidateSet('all','tongji_final_report','tongji_project_plan')]
    [string]$Target = 'all'
)
$ErrorActionPreference = 'Stop'
$reportRoot = $PSScriptRoot
$projectRoot = Split-Path $reportRoot -Parent
$localTeX = Join-Path $projectRoot '.tools\miktex\miktex\bin\x64'
if (Test-Path (Join-Path $localTeX 'xelatex.exe')) {
    $env:Path = $localTeX + ';' + $env:Path
    $env:MIKTEX_USERCONFIG = Join-Path $projectRoot '.tools\miktex-config'
    $env:MIKTEX_USERDATA = Join-Path $projectRoot '.tools\miktex-data'
    $env:MIKTEX_USERINSTALL = Join-Path $projectRoot '.tools\miktex'
}
$bibtexCommand = if (Get-Command bibtex8 -ErrorAction SilentlyContinue) { 'bibtex8' } else { 'bibtex' }
$env:MIKTEX_ENABLEINSTALLER = 't'
$env:PYTHONIOENCODING = 'utf-8'
$reportNames = if ($Target -eq 'all') {
    @('tongji_final_report','tongji_project_plan')
} else { @($Target) }
foreach ($reportName in $reportNames) {
    Push-Location (Join-Path $reportRoot $reportName)
    try {
        & xelatex -enable-installer -include-directory=style -interaction=nonstopmode -halt-on-error -file-line-error main.tex *> build-1.log
        if ($LASTEXITCODE -ne 0) { Get-Content build-1.log -Tail 35; throw "$($reportName): XeLaTeX pass 1 failed" }
        & $bibtexCommand main *> build-bibtex.log
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 1) { Get-Content build-bibtex.log -Tail 35; throw "$($reportName): BibTeX failed" }
        foreach ($pass in @(2,3)) {
            & xelatex -include-directory=style -interaction=nonstopmode -halt-on-error -file-line-error main.tex *> "build-$pass.log"
            if ($LASTEXITCODE -ne 0) { Get-Content "build-$pass.log" -Tail 35; throw "$($reportName): XeLaTeX failed" }
        }
        if (Select-String -Path main.log -Pattern 'Please \(re\)run BibTeX' -Quiet) {
            & $bibtexCommand main *> build-bibtex-2.log
            if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 1) { throw "$($reportName): final BibTeX pass failed" }
            foreach ($pass in @(4,5)) {
                & xelatex -include-directory=style -interaction=nonstopmode -halt-on-error -file-line-error main.tex *> "build-$pass.log"
                if ($LASTEXITCODE -ne 0) { throw "$($reportName): final XeLaTeX pass failed" }
            }
        }
        Write-Output "$reportName compiled"
        Select-String -Path main.log -Pattern 'Output written|Overfull|Float too large|undefined references|Missing character'
    } finally { Pop-Location }
}
