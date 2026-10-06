@echo off
set "QNN_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.45.40.260406"
set "LIB=%QNN_SDK_ROOT%\lib\aarch64-windows-msvc"
set "PATH=%LIB%;%QNN_SDK_ROOT%\bin\aarch64-windows-msvc;%PATH%"
set "ADSP_LIBRARY_PATH=%LIB%"
cd /d C:\AI\npu-work
if not exist out_htp245 mkdir out_htp245
"%QNN_SDK_ROOT%\bin\aarch64-windows-msvc\qnn-net-run.exe" ^
  --backend "%LIB%\QnnHtp.dll" ^
  --dlc_path C:\AI\npu-work\ve245_int8.dlc ^
  --input_list C:\AI\npu-work\inputs\input_list.txt ^
  --output_dir C:\AI\npu-work\out_htp245 ^
  %*
echo EXITCODE=%ERRORLEVEL%