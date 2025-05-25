# package_deployment.ps1
# Packages Fixacar SKU Finder app, trainer, models, and required data for deployment.

$deployDir = "Fixacar_Deploy"
$distDir = "dist"
$srcDir = "src"
$modelsDir = "models"
$dataDir = "data"
$sourceFilesDir = "Source_Files"
$requirementsFile = "requirements.txt"

Write-Host "=== Packaging Fixacar SKU Finder Deployment ==="

# Clean previous deployment
if (Test-Path $deployDir) { Remove-Item $deployDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $deployDir | Out-Null

# Copy executables
Copy-Item "$distDir\Fixacar_SKU_Finder.exe" $deployDir
Copy-Item "$distDir\Fixacar_SKU_Trainer.exe" $deployDir

# Copy models (including all encoders/tokenizer)
Copy-Item $modelsDir $deployDir\models -Recurse

# Copy data (e.g., Maestro.xlsx, databases)
Copy-Item $dataDir $deployDir\data -Recurse

# Copy Source_Files (if needed for operation)
Copy-Item $sourceFilesDir $deployDir\Source_Files -Recurse

# Copy requirements.txt for reference
Copy-Item $requirementsFile $deployDir

# Optionally copy README or deployment instructions if present
if (Test-Path "README.md") { Copy-Item "README.md" $deployDir }

Write-Host "=== Packaging Complete! ==="
Write-Host "Deployment directory: $deployDir"
