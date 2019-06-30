@echo off

cd xgboost
git submodule init
git submodule update
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
msbuild xgboost.sln /m /p:Configuration=Release /p:Platform=x64

if %errorlevel% neq 0 exit /b 1

cd ../python-package
python setup.py install

if %errorlevel% neq 0 exit /b 1

cd ../..

cd alphamind\pfopt
git submodule init
git submodule update

set BUILD_TEST=OFF
call build_windows.bat

if %errorlevel% neq 0 exit /b 1

cd ../..

@echo on