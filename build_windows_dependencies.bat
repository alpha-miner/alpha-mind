@echo off

cd alphamind\pfopt

call build_windows.bat

if %errorlevel% neq 0 exit /b 1

cd ../..

@echo on