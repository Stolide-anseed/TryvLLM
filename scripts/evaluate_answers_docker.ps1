param(
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]]$EvaluationArgs,
    [string]$Image = "tryvllm:dev"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $projectRoot ".env"

$dockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all",
    "--ipc=host",
    "--workdir", "/app"
)

if (Test-Path -LiteralPath $envFile) {
    $dockerArgs += @("--env-file", $envFile)
}

$dockerArgs += @(
    "-e", "LLM_QDRANT_URL=http://host.docker.internal:6333",
    "-v", "${projectRoot}:/app",
    "-v", "hf-cache:/root/.cache/huggingface",
    "-v", "vllm-cache:/root/.cache/vllm",
    "-v", "fastembed-cache:/root/.cache/fastembed",
    "--entrypoint", "python3",
    $Image,
    "-m", "scripts.evaluate_answers"
)
$dockerArgs += $EvaluationArgs

& docker @dockerArgs
exit $LASTEXITCODE
