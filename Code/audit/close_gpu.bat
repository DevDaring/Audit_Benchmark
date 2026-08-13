@echo off
REM close_gpu.bat -- one-click teardown of the Akash GPU lease.
REM An idle A100 bills about $3.75/hour and the container does not exit on its own.
cd /d "%~dp0"
echo Checking lease status...
python TIST/deploy_tist.py --status
echo.
echo Closing lease...
python TIST/deploy_tist.py --close
echo.
echo Done. If it said success, billing has stopped.
pause
