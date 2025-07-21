@echo off 
echo Testing VIN Training... 
call train_vin_auto.bat 
echo. 
echo Testing SKU Training... 
call train_sku_auto.bat 
echo All tests completed! 
pause 
