@echo off
REM WAM Environment Setup Script for Windows
REM This script creates a conda environment and installs all dependencies

echo 🚀 Setting up WAM environment...

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda is not installed. Please install Anaconda or Miniconda first.
    echo Visit: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr "wam" >nul
if %errorlevel% equ 0 (
    echo ⚠️  Environment 'wam' already exists.
    set /p choice="Do you want to remove it and create a fresh one? (y/N): "
    if /i "%choice%"=="y" (
        echo 🗑️  Removing existing environment...
        conda env remove -n wam -y
    ) else (
        echo ✅ Using existing environment. Activate it with: conda activate wam
        pause
        exit /b 0
    )
)

REM Create new environment
echo 📦 Creating conda environment 'wam' with Python 3.9...
conda create -n wam python=3.9 -y

REM Activate environment
echo 🔧 Activating environment...
call conda activate wam

REM Install PyTorch
echo 🔥 Installing PyTorch...
REM Check if CUDA is available (simplified check)
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo 🎮 CUDA detected, installing PyTorch with CUDA support...
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
) else (
    echo 💻 Installing PyTorch for CPU...
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
)

REM Install remaining dependencies
echo 📚 Installing other dependencies...
pip install -r requirements.txt

REM Verify installation
echo ✅ Verifying installation...
python -c "import torch; import ptwt; import cv2; print('🎉 All dependencies installed successfully!')"

echo.
echo 🎊 Environment setup complete!
echo.
echo To activate the environment, run:
echo   conda activate wam
echo.
echo To start using WAM:
echo   jupyter notebook example.ipynb
echo.
echo Happy coding! 🚀
pause 