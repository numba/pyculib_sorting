cd %RECIPE_DIR%\..
call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64
python build_sorting_libs.py

mkdir %PREFIX%\DLLs
copy %RECIPE_DIR%\..\lib\*.dll %PREFIX%\DLLs
