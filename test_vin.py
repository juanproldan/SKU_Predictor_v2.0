import sys
sys.path.append('src')

from train_vin_predictor import extract_vin_features_production, decode_year

# Test VIN
test_vin = "3MVDM2W7AML103902"

print(f"Testing VIN: {test_vin}")
print(f"Length: {len(test_vin)}")

# Extract features
features = extract_vin_features_production(test_vin)
if features:
    print("\n=== VIN Features ===")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Decode year
    year = decode_year(features['year_code'])
    print(f"\n=== Decoded Year ===")
    print(f"  Year code '{features['year_code']}' -> {year}")
    
    print(f"\n=== VIN Breakdown ===")
    print(f"  WMI (World Manufacturer Identifier): {features['wmi']} (positions 1-3)")
    print(f"  VDS (Vehicle Descriptor Section): {features['vds']} (positions 4-8)")
    print(f"  VDS Full: {features['vds_full']} (positions 4-9)")
    print(f"  Year Code: {features['year_code']} (position 10)")
    print(f"  Plant Code: {features['plant_code']} (position 11)")
    print(f"  Serial Number: {test_vin[11:]} (positions 12-17)")
else:
    print("❌ Failed to extract features from VIN")

# Check what WMI codes exist in our database for Mazda
print(f"\n=== Database Check ===")
import sqlite3
conn = sqlite3.connect('Source_Files/processed_consolidado.db')
cursor = conn.cursor()

# Check if this WMI exists in our database
cursor.execute("SELECT DISTINCT maker, COUNT(*) FROM processed_consolidado WHERE vin_number LIKE '3MV%' GROUP BY maker")
wmi_results = cursor.fetchall()
print(f"VINs starting with '3MV' in database:")
for maker, count in wmi_results:
    print(f"  {maker}: {count} records")

# Check all Mazda WMI codes in database
cursor.execute("SELECT DISTINCT SUBSTR(vin_number, 1, 3) as wmi, COUNT(*) FROM processed_consolidado WHERE UPPER(maker) = 'MAZDA' AND vin_number IS NOT NULL GROUP BY wmi ORDER BY COUNT(*) DESC")
mazda_wmis = cursor.fetchall()
print(f"\nMazda WMI codes in database:")
for wmi, count in mazda_wmis:
    print(f"  {wmi}: {count} records")

# Check specific VDS combinations for this WMI
cursor.execute("SELECT DISTINCT SUBSTR(vin_number, 4, 6) as vds_full, series, COUNT(*) FROM processed_consolidado WHERE SUBSTR(vin_number, 1, 3) = '3MV' AND UPPER(maker) = 'MAZDA' GROUP BY vds_full, series ORDER BY COUNT(*) DESC LIMIT 10")
vds_combinations = cursor.fetchall()
print(f"\nTop VDS combinations for WMI '3MV':")
for vds, series, count in vds_combinations:
    print(f"  VDS: {vds}, Series: {series}, Count: {count}")

# Check if our specific combination exists
cursor.execute("SELECT DISTINCT series, COUNT(*) FROM processed_consolidado WHERE SUBSTR(vin_number, 1, 3) = '3MV' AND SUBSTR(vin_number, 4, 6) = 'DM2W7A' AND UPPER(maker) = 'MAZDA' GROUP BY series")
specific_combo = cursor.fetchall()
print(f"\nOur specific combination (WMI=3MV, VDS=DM2W7A):")
if specific_combo:
    for series, count in specific_combo:
        print(f"  Series: {series}, Count: {count}")
else:
    print("  ❌ No records found for this exact combination")

# Check similar VDS patterns
cursor.execute("SELECT DISTINCT SUBSTR(vin_number, 4, 6) as vds_full, series, COUNT(*) FROM processed_consolidado WHERE SUBSTR(vin_number, 1, 3) = '3MV' AND SUBSTR(vin_number, 4, 6) LIKE 'DM2W7%' AND UPPER(maker) = 'MAZDA' GROUP BY vds_full, series ORDER BY COUNT(*) DESC")
similar_vds = cursor.fetchall()
print(f"\nSimilar VDS patterns (DM2W7*):")
for vds, series, count in similar_vds:
    print(f"  VDS: {vds}, Series: {series}, Count: {count}")

conn.close()

# Test the actual model prediction
print(f"\n=== Testing Model Prediction ===")
try:
    import joblib
    import pandas as pd

    # Load the models
    MODEL_DIR = 'Models'
    model_series = joblib.load(f'{MODEL_DIR}/series_model.joblib')
    encoder_x_series = joblib.load(f'{MODEL_DIR}/series_encoder_x.joblib')
    encoder_y_series = joblib.load(f'{MODEL_DIR}/series_encoder_y.joblib')

    # Test our specific VIN
    wmi = features['wmi']
    vds_full = features['vds_full']

    print(f"Testing: WMI='{wmi}', VDS_FULL='{vds_full}'")

    # Create DataFrame for prediction
    series_df = pd.DataFrame([[wmi, vds_full]], columns=['wmi', 'vds_full'])
    print(f"Input DataFrame: {series_df}")

    # Transform features
    series_features_encoded = encoder_x_series.transform(series_df)
    print(f"Encoded features: {series_features_encoded}")

    # Check if any feature was marked as unknown (-1)
    if -1 in series_features_encoded[0]:
        print("❌ One or more features marked as UNKNOWN by encoder")
        print(f"   WMI '{wmi}' encoded as: {series_features_encoded[0][0]}")
        print(f"   VDS_FULL '{vds_full}' encoded as: {series_features_encoded[0][1]}")

        # Check what values the encoder knows
        print("\nKnown WMI values in encoder:")
        wmi_categories = encoder_x_series.categories_[0]  # First column (wmi)
        print(f"  Total WMI categories: {len(wmi_categories)}")
        if wmi in wmi_categories:
            print(f"  ✅ '{wmi}' is known")
        else:
            print(f"  ❌ '{wmi}' is NOT known")
            # Show similar WMIs
            similar_wmis = [w for w in wmi_categories if w.startswith('3M')]
            print(f"  Similar WMIs: {similar_wmis[:10]}")

        print("\nKnown VDS_FULL values in encoder:")
        vds_categories = encoder_x_series.categories_[1]  # Second column (vds_full)
        print(f"  Total VDS_FULL categories: {len(vds_categories)}")
        if vds_full in vds_categories:
            print(f"  ✅ '{vds_full}' is known")
        else:
            print(f"  ❌ '{vds_full}' is NOT known")
            # Show similar VDS
            similar_vds = [v for v in vds_categories if v.startswith('DM2W7')]
            print(f"  Similar VDS: {similar_vds[:10]}")
    else:
        # Make prediction
        series_pred_encoded = model_series.predict(series_features_encoded)
        print(f"Prediction encoded: {series_pred_encoded}")

        if series_pred_encoded[0] != -1:
            series_result = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0]
            print(f"✅ Predicted series: '{series_result}'")
        else:
            print("❌ Model returned unknown prediction")

except Exception as e:
    print(f"❌ Error testing model: {e}")
    import traceback
    traceback.print_exc()

# Test the fallback strategy
print(f"\n=== Testing Fallback Strategy ===")
try:
    import sys
    sys.path.append('src')
    from main_app import FixacarApp

    # Create a dummy app instance to test the method
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    app = FixacarApp(root)

    # Test the fallback method
    wmi = "3MV"
    maker = "Mazda"
    fallback_series = app.get_most_common_series_for_wmi(wmi, maker)
    print(f"Fallback series for WMI '{wmi}' and maker '{maker}': '{fallback_series}'")

    root.destroy()

except Exception as e:
    print(f"❌ Error testing fallback: {e}")
    import traceback
    traceback.print_exc()

# Test the database search that's causing the error
print(f"\n=== Testing Database Search ===")
try:
    import sqlite3
    conn = sqlite3.connect('Source_Files/processed_consolidado.db')
    cursor = conn.cursor()

    # Test the actual dual matching strategy queries (with the fix)
    print("Testing original_vs_db query (with None filter):")
    cursor.execute("""
        SELECT referencia, COUNT(*) as frequency, 'original_vs_db' as match_type
        FROM processed_consolidado
        WHERE LOWER(maker) = LOWER(?) AND model = ? AND LOWER(series) = LOWER(?) AND LOWER(descripcion) = LOWER(?)
        AND referencia IS NOT NULL AND referencia != '' AND referencia != 'None' AND referencia != 'UNKNOWN'
        GROUP BY referencia
        ORDER BY COUNT(*) DESC
    """, ('Mazda', '2021', 'CX-30', 'PUERTA DELANTERA DERECHA'))
    original_results = cursor.fetchall()
    print(f"  Original results: {original_results}")

    print("Testing normalized_vs_db query (with None filter):")
    cursor.execute("""
        SELECT referencia, COUNT(*) as frequency, 'normalized_vs_db' as match_type
        FROM processed_consolidado
        WHERE LOWER(maker) = LOWER(?) AND model = ? AND LOWER(series) = LOWER(?) AND LOWER(descripcion) = LOWER(?)
        AND referencia IS NOT NULL AND referencia != '' AND referencia != 'None' AND referencia != 'UNKNOWN'
        GROUP BY referencia
        ORDER BY COUNT(*) DESC
    """, ('Mazda', '2021', 'CX-30', 'puerta delantera derecha'))
    normalized_results = cursor.fetchall()
    print(f"  Normalized results: {normalized_results}")

    # Combine results like the actual code does
    all_results = original_results + normalized_results

    print(f"Combined results: {all_results}")
    print(f"Number of combined results: {len(all_results)}")

    if all_results:
        print("Testing tuple unpacking:")
        for i, result in enumerate(all_results):
            print(f"  Result {i}: {result} (length: {len(result)})")
            try:
                referencia, frequency, match_type = result
                print(f"    ✅ Unpacked: referencia='{referencia}', frequency={frequency}, match_type='{match_type}'")
            except ValueError as e:
                print(f"    ❌ Unpacking error: {e}")

        # Test the consensus logic conversion
        print("\nTesting consensus logic conversion:")
        try:
            exact_results_for_consensus = [(referencia, frequency) for referencia, frequency, match_type in all_results]
            print(f"  ✅ Consensus conversion successful: {exact_results_for_consensus}")
        except ValueError as e:
            print(f"  ❌ Consensus conversion error: {e}")

    conn.close()

except Exception as e:
    print(f"❌ Error testing database search: {e}")
    import traceback
    traceback.print_exc()
