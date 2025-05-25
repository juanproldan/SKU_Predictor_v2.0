# build_executables.ps1
# Builds Fixacar SKU Finder GUI and model training workflow as Windows executables using PyInstaller.

# Ensure script stops on error
$ErrorActionPreference = "Stop"

Write-Host "=== Building Fixacar SKU Finder Executables ==="

# Set paths
$srcDir = "src"
$mainApp = "main_app.py"
$trainScript = "train_sku_nn_predictor_pytorch_optimized.py"
$distDir = "dist"
$buildDir = "build"
$specDir = "spec"

# Clean previous builds
if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }
if (Test-Path $specDir) { Remove-Item $specDir -Recurse -Force }

# Create spec directory for PyInstaller specs
New-Item -ItemType Directory -Force -Path $specDir | Out-Null

# Build main_app.exe
Write-Host "`n--- Building main_app.exe ---"
pyinstaller --noconfirm --onefile --windowed --name "Fixacar_SKU_Finder" `
    --distpath $distDir `
    --workpath $buildDir `
    --add-data ".\models;models" `
    --add-data ".\data;data" `
    --add-data ".\Source_Files;Source_Files" `
    "$srcDir\$mainApp"

# Build train_sku_nn_predictor_pytorch_optimized.exe
Write-Host "`n--- Building train_sku_nn_predictor_pytorch_optimized.exe ---"
pyinstaller --noconfirm --onefile --console --name "Fixacar_SKU_Trainer" `
    --distpath $distDir `
    --workpath $buildDir `
    --add-data ".\models;models" `
    --add-data ".\data;data" `
    --add-data ".\Source_Files;Source_Files" `
    "$srcDir\$trainScript"

Write-Host "`n=== Build Complete! ==="
Write-Host "Executables are in the 'dist' folder."
