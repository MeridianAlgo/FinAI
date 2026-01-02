@echo off
REM Initialize Fin.AI v2 model locally and upload to Hugging Face

echo 🚀 Fin.AI v2 Initialization Script
echo.

REM Check if HF_TOKEN is set
if "%HF_TOKEN%"=="" (
    echo ❌ Error: HF_TOKEN environment variable not set
    echo.
    echo Please set your Hugging Face token:
    echo   set HF_TOKEN=your_token_here
    echo.
    echo Get your token from: https://huggingface.co/settings/tokens
    exit /b 1
)

echo ✓ HF_TOKEN found
echo.

REM Run the initialization script
python scripts/init_v2_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Success! Your v2 model is now on Hugging Face
    echo.
    echo 🎯 Next steps:
    echo    1. Commit and push any local changes
    echo    2. The next GitHub Actions training run will use v2
    echo    3. Or trigger a manual training run from GitHub Actions
) else (
    echo.
    echo ❌ Initialization failed. Check the error messages above.
    exit /b 1
)
