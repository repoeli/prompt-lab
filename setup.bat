@echo off
REM setup.bat - Professional Prompt Lab Setup Script for Windows
REM Run this script to set up your prompt lab environment

echo 🚀 Prompt Lab Professional Setup
echo ================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    echo 💡 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
if exist "requirements-pinned.txt" (
    pip install -r requirements-pinned.txt
) else (
    pip install -r requirements.txt
)

REM Create .env if it doesn't exist
if not exist ".env" (
    echo 🔐 Creating .env file...
    copy .env.example .env
    echo ⚠️  Please edit .env and add your OPENAI_API_KEY
) else (
    echo ✅ .env file already exists
)

REM Create data directories
if not exist "data\evaluations" mkdir data\evaluations
if not exist "data\exports" mkdir data\exports

REM Run smoke test
echo 🧪 Running smoke test...
python smoke_test.py

if errorlevel 1 (
    echo ❌ Smoke test failed. Please check your setup.
    echo 💡 Make sure you've added your OPENAI_API_KEY to .env
    pause
    exit /b 1
)

echo.
echo 🎉 Setup Complete!
echo ===============
echo.
echo Next steps:
echo 1. Edit .env and add your OPENAI_API_KEY
echo 2. Run: jupyter lab notebooks\01_foundations.ipynb
echo 3. Start experimenting!
echo.
pause
