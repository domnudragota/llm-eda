$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $ProjectRoot 'lwow'
$Commit = '455ca9d475b4c3a55d48952689ba6db18cb82574'
function Invoke-LwowGit {
    & git -c "safe.directory=$($Dest.Replace('\', '/'))" -C $Dest @args
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $args" }
}
$Paths = @(
    '/README.md', '/LICENSE*',
    '/reproducibility/FA_Functions.py',
    '/reproducibility/FA_spreadr.r',
    '/reproducibility/FA_build_Networks.py',
    '/reproducibility/FA_analyses_LDT_Gender.py',
    '/reproducibility/data/LDT_analyses/FA_matrices/Mistral_gender.csv',
    '/reproducibility/data/LDT_analyses/FA_matrices/Humans_gender.csv',
    '/reproducibility/data/LDT_analyses/FA_matrices/Llama3_gender.csv',
    '/reproducibility/data/LDT_analyses/FA_matrices/Haiku_gender.csv',
    '/reproducibility/data/LDT_analyses/gender_targets.csv',
    '/reproducibility/data/LDT_analyses/gender_primes.csv',
    '/reproducibility/data/LDT_analyses/LDT_50_triplets.csv',
    '/reproducibility/data/LDT_analyses/Statistical_tests/',
    '/reproducibility/data/summary_tables/',
    '/LWOW_datasets/graphs/edge_lists/FA_Mistral_edgelist.csv',
    '/LWOW_datasets/graphs/edge_lists/FA_Llama3_edgelist.csv',
    '/LWOW_datasets/graphs/edge_lists/FA_Haiku_edgelist.csv'
)
if (Test-Path -LiteralPath $Dest) {
    $Head = Invoke-LwowGit rev-parse HEAD
    if ($Head -ne $Commit) { throw "Existing revision is $Head; expected $Commit. No checkout changed." }
    if (Invoke-LwowGit status --porcelain --untracked-files=no) {
        throw 'Tracked researcher files have local edits; no checkout changed.'
    }
    $Missing = @($Paths | Where-Object { $_ -notmatch '[*]' -and -not (Test-Path -LiteralPath (Join-Path $Dest $_.TrimStart('/'))) })
    if ($Missing.Count -gt 0) { Invoke-LwowGit sparse-checkout add @Missing }
} else {
    & git clone --config core.autocrlf=false --filter=blob:none --no-checkout https://github.com/LLMWorldOfWords/LWOW.git $Dest
    if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
    Invoke-LwowGit sparse-checkout set --no-cone @Paths
    Invoke-LwowGit checkout --detach $Commit
}
Write-Host "LWOW repo synced locally."
