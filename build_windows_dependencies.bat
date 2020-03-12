@echo off

cd alphamind\pfopt
git submodule init
git submodule update

set BUILD_TEST=OFF
call build_windows.bat

if %errorlevel% neq 0 exit /b 1

cd ../..

@echo on