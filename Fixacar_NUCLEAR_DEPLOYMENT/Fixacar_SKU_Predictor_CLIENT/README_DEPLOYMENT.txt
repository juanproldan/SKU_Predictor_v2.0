FIXACAR SKU PREDICTOR - CLIENT DEPLOYMENT PACKAGE
=================================================

This folder contains the complete client deployment package for the Fixacar SKU Predictor system.

MAIN APPLICATION:
-----------------
• Fixacar_SKU_Predictor.exe - Main application for SKU predictions
• Fixacar_SKU_Predictor.bat - Launcher with diagnostic information

SETUP TOOLS:
-------------
• 1. Fixacar_Consolidado_Downloader.exe - Downloads latest consolidado.json data
• 2. Fixacar_Data_Processor.exe - Processes consolidado.json into database
• 3. Fixacar_VIN_Trainer.exe - Trains VIN prediction models
• 4. Fixacar_SKU_Trainer.exe - Trains SKU prediction models

FOLDERS:
--------
• Source_Files/ - Contains all data files (Excel, JSON, Database)
• models/ - Contains all trained AI models
• logs/ - Application logs and diagnostics

DEPLOYMENT INSTRUCTIONS:
------------------------
1. Copy this entire folder to the client computer
2. Run setup tools (1-4) in order to initialize the system
3. Run Fixacar_SKU_Predictor.exe for daily operations

REQUIREMENTS:
-------------
• Windows 10/11
• No additional software installation required
• All dependencies are bundled in the executables

SELF-CONTAINED:
---------------
• All files read from and written to this folder only
• No external dependencies
• Portable - can be moved between computers

SUPPORT:
--------
For technical support, contact the development team.
