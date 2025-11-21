# всегда работать из директории скрипта
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Шаг 1: виртуальное окружение"


if (-not (Test-Path "venv")) {
    Write-Host "Создаю venv..."
    python -m venv venv
} else {
    Write-Host "venv уже существует, пропускаю создание"
}

$pythonExe = Join-Path $scriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "ОШИБКА: $pythonExe не найден"
    Write-Host "Проверь, что Python установлен и venv создан корректно."
    exit 1
}

Write-Host "Шаг 2: установка зависимостей"

if (-not (Test-Path "requirements.txt")) {
    Write-Host "ОШИБКА: requirements.txt не найден в корне проекта."
    exit 1
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r "requirements.txt"

Write-Host "Шаг 3: подготовка папки llama.cpp"

$llamaDir = Join-Path $scriptDir "llama.cpp"

if (-not (Test-Path $llamaDir)) {
    Write-Host "Папка llama.cpp не найдена, создаю..."
    New-Item -ItemType Directory -Path $llamaDir | Out-Null
} else {
    Write-Host "llama.cpp уже существует, пропускаю создание"
}

if (-not (Test-Path $llamaDir)) {
    Write-Host "ОШИБКА: не удалось создать папку llama.cpp."
    exit 1
}

Write-Host "Шаг 4: распаковка Ilama.cpp.zip в llama.cpp"

$zipName = "Ilama.cpp.zip"
$zipPath = Join-Path $scriptDir $zipName

if (Test-Path $zipPath) {
    Write-Host "Найден ZIP: $zipPath"
    Write-Host "Распаковываю ZIP в llama.cpp..."

    # временная папка для распаковки
    $tempUnpack = Join-Path $scriptDir "llama_temp_unpack"

    if (Test-Path $tempUnpack) {
        Remove-Item $tempUnpack -Recurse -Force
    }

    Expand-Archive -Path $zipPath -DestinationPath $tempUnpack -Force

    Copy-Item (Join-Path $tempUnpack "*") (Join-Path $scriptDir "llama.cpp") -Recurse -Force

    Remove-Item $tempUnpack -Recurse -Force

    try {
    Remove-Item -LiteralPath $zipPath -Force
    Write-Host "ZIP удалён: $zipPath"
    } catch {
        Write-Host "НЕ УДАЛОСЬ удалить ZIP: $($_.Exception.Message)"
    }


} else {
    Write-Host "ПРЕДУПРЕЖДЕНИЕ: $zipName не найден."
}


