@echo off
REM Script to install and integrate ns3-gym opengym module (Windows)

setlocal enabledelayedexpansion

echo Installing ns3-gym opengym module...

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "NS3_DIR=%SCRIPT_DIR%"

REM Check if ns3-gym directory exists
if not exist "%NS3_DIR%ns3-gym" (
    echo Cloning ns3-gym repository...
    cd /d "%NS3_DIR%"
    git clone https://github.com/tkn-tub/ns3-gym.git
)

REM Copy opengym module to src directory
if exist "%NS3_DIR%ns3-gym\opengym" (
    echo Copying opengym module to src\...
    xcopy /E /I /Y "%NS3_DIR%ns3-gym\opengym" "%NS3_DIR%src\opengym"
    echo opengym module copied successfully!
) else (
    echo Error: opengym module not found in ns3-gym directory
    echo Please check if ns3-gym was cloned correctly
    exit /b 1
)

REM Build opengym module
echo Building opengym module...
cd /d "%NS3_DIR%"
waf configure
waf build

echo opengym module installation complete!
echo.
echo Next steps:
echo 1. Make sure 'opengym' is added to dependencies in src/point-to-point/wscript
echo 2. Rebuild the project: waf build

