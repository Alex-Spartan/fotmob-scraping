import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from modules import scrape as sc

# sc.scrapeData("https://www.fotmob.com/en-GB/teams/8535/overview/fiorentina")

df = pd.read_excel("Inter.xlsx")

# Select relevant columns (excluding non-numeric)
stats_columns = [col for col in df.columns if "Home" in col or "Away" in col]
print(stats_columns)
exit()
df = df[stats_columns]

# Convert stats to numeric (handle % if needed)
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# Prepare data for ML
sequence_length = 5  # Use last 5 matches to predict next match

X, y = [], []

for i in range(len(df) - sequence_length):
    X.append(df.iloc[i:i+sequence_length].values.flatten())  # Flatten stats of 5 matches
    y.append(df.iloc[i+sequence_length].values)  # Next match stats

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data (normalize)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Predict next match stats (use last 5 matches)
next_match_input = df.iloc[-sequence_length:].values.flatten().reshape(1, -1)
next_match_input = scaler.transform(next_match_input)

predicted_stats = model.predict(next_match_input)
print("Predicted next match stats:", predicted_stats)