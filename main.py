import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from modules import scrape as sc

sc.scrapeData("https://www.fotmob.com/en-GB/teams/8633/fixtures/real-madrid?before=4506988")
exit()
# Read the Excel file
df = pd.read_excel("real-madrid.xlsx")
df.to_csv("real-madrid.csv", index=False)


# Select relevant numeric columns
stats_columns = [col for col in df.columns if ("Home" in col or "Away" in col) and col not in ["Home Team", "Away Team"]]
df = df[stats_columns]

# Convert stats to numeric, handling errors
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
# Prepare data for ML
sequence_length = 1  # Number of past matches to consider

X, y = [], []

if len(df) > sequence_length:
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length].values.flatten())  # Flatten match stats
        y.append(df.iloc[i+sequence_length].values)  # Next match stats

    X, y = np.array(X), np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize using MinMaxScaler (better for varied football stats)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a separate Gradient Boosting model for each target variable
    models = []
    y_pred_list = []
    

    for i in range(y_train.shape[1]):
        model = SVR(kernel='rbf')  # 'rbf' kernel captures non-linearity
        model.fit(X_train, y_train[:, i])
        models.append(model)


        # Predict for test set
        y_pred_list.append(model.predict(X_test))

    # Combine predictions for all stats
    y_pred = np.column_stack(y_pred_list)

    # Evaluate model
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')  # Averaging R² across stats
    print(f"R² Score: {r2:.4f}")

    # Predict next match stats
    next_match_input = df.iloc[-sequence_length:].values.flatten().reshape(1, -1)
    next_match_input = scaler.transform(next_match_input)

    predicted_stats = np.column_stack([model.predict(next_match_input) for model in models])
    predicted_df = pd.DataFrame(predicted_stats, columns=df.columns)

    # Display predicted statistics with their names
    print("Predicted next match stats:")
    for col in predicted_df.columns:
        print(f"{col}: {predicted_df[col].values[0]:.2f}")

else:
    print(f"Not enough data to create sequences of length {sequence_length}")
