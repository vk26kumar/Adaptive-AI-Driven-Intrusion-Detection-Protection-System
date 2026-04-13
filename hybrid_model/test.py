import pandas as pd
from predict import predict_intrusion

# Load small sample (use your dataset)
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# FIX column names (important)
df.columns = df.columns.str.strip()

# Now drop label
df = df.drop(columns=["Label"])

# Take small sample (for fast test)
df_sample = df.head(10)

# Run prediction
results = predict_intrusion(df_sample)

print(results)