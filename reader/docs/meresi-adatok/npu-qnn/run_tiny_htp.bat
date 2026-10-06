@echo off
set "QNN_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.45.40.260406"
set "LIB=%QNN_SDK_ROOT%\lib\aarch64-windows-msvc"
set "PATH=%LIB%;%QNN_SDK_ROOT%\bin\aarch64-windows-msvc;%PATH%"
set "ADSP_LIBRARY_PATH=%LIB%"
cd /d C:\AI\npu-work
if not exist out_tiny mkdir out_tiny
"%QNN_SDK_ROOT%\bin\aarch64-windows-msvc\qnn-net-run.exe" ^
  --backend "%LIB%\QnnHtp.dll" ^
  --dlc_path C:\AI\npu-work\tiny\tiny245.dlc ^
  --input_list C:\AI\npu-work\tiny\tiny_list.txt ^
  --output_dir C:\AI\npu-work\out_tiny ^
  %*
echo EXITCODE=%ERRORLEVEL%