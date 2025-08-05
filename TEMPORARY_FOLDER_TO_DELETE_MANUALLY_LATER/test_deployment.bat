@echo off
echo.
echo ========================================
echo   FIXACAR DEPLOYMENT TESTING SUITE
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo 🔬 Running Isolated Dependency Tests...
echo.
python isolated_dependency_test.py

echo.
echo ========================================
echo   DEPLOYMENT TEST COMPLETE
echo ========================================
echo.
echo ✅ If all tests passed, executables are ready for client deployment!
echo ⚠️  If any tests failed, rebuild the failing executables.
echo.
pause
