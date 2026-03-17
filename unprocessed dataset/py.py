import pandas as pd

# Load JSON
df = pd.read_json("github.json")

# Flatten nested JSON (very important for your dataset)
df_flat = pd.json_normalize(df.to_dict(orient="records"))

# Save to CSV
df_flat.to_csv("GitHub_10k.csv", index=False)

print("✅ JSON converted to CSV successfully!")
print("Shape:", df_flat.shape)