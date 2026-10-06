@echo off
set "QAIRT_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.50.40.260831"
set "GENIEX_HTP=C:\Users\istva\AppData\Local\GenieX CLI\qairt\htp-files"
set "PATH=%GENIEX_HTP%;%QAIRT_SDK_ROOT%\lib\arm64x-windows-msvc;%QAIRT_SDK_ROOT%\bin\arm64x-windows-msvc;%PATH%"
set "ADSP_LIBRARY_PATH=%GENIEX_HTP%;%QAIRT_SDK_ROOT%\lib\arm64x-windows-msvc"
cd /d C:\AI\npu-work
if not exist out_htp mkdir out_htp
"%QAIRT_SDK_ROOT%\bin\arm64x-windows-msvc\qairt-net-run.exe" ^
  --input_dlc C:\AI\npu-work\vector_estimator_int8.dlc ^
  --input_list C:\AI\npu-work\inputs\input_list.txt ^
  --use_htp ^
  --output_dir C:\AI\npu-work\out_htp ^
  %*
echo EXITCODE=%ERRORLEVEL%