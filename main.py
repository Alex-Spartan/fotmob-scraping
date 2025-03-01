import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from modules import scrape as sc

# sc.scrapeData("https://www.fotmob.com/en-GB/teams/8535/fixtures/fiorentina?before=4535434")
df = pd.read_excel("Fiorentina.xlsx")
df.to_csv("Fiorentina.csv", index=False)


stats_columns = [col for col in df.columns if ("Home" in col or "Away" in col) and col not in ["Home Team", "Away Team"]]
df = df[stats_columns]

df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
sequence_length = 1

X, y = [], []

if len(df) > sequence_length:
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length].values.flatten())
        y.append(df.iloc[i+sequence_length].values)

    X, y = np.array(X), np.array(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = []
    y_pred_list = []
    

    for i in range(y_train.shape[1]):
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train[:, i])
        models.append(model)


        y_pred_list.append(model.predict(X_test))

    y_pred = np.column_stack(y_pred_list)

    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    exit()

    next_match_input = df.iloc[-sequence_length:].values.flatten().reshape(1, -1)
    next_match_input = scaler.transform(next_match_input)

    predicted_stats = np.column_stack([model.predict(next_match_input) for model in models])
    predicted_df = pd.DataFrame(predicted_stats, columns=df.columns)

    print("Predicted next match stats:")
    for col in predicted_df.columns:
        print(f"{col}: {predicted_df[col].values[0]:.2f}")

else:
    print(f"Not enough data to create sequences of length {sequence_length}")