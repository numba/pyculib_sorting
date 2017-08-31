cd %RECIPE_DIR%\..
call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64
%PYTHON% build_sorting_libs.py
if errorlevel 1 exit 1

mkdir %PREFIX%\DLLs
copy %RECIPE_DIR%\..\lib\*.dll %PREFIX%\DLLs
