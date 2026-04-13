import os

file_path = "hybrid_model/predict.py"

# Read file
with open(file_path, "r") as f:
    content = f.read()

# Replace paths
content = content.replace('load_model("autoencoder.keras")',
                          'load_model("hybrid_model/autoencoder.keras")')

content = content.replace('"xgb_model.pkl"',
                          '"hybrid_model/xgb_model.pkl"')

content = content.replace('"scaler.pkl"',
                          '"hybrid_model/scaler.pkl"')

content = content.replace('"threshold.json"',
                          '"hybrid_model/threshold.json"')

content = content.replace('"selected_columns.pkl"',
                          '"hybrid_model/selected_columns.pkl"')

# Write back
with open(file_path, "w") as f:
    f.write(content)

print("Paths fixed successfully!")