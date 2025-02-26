import pandas as pd
import re

# Example data
data = [
    {'Match': 'Braga vs Lazio', 'Home Team': 'Lazio', 'Away Team': 'Braga', 'Home Score': 0, 'Away Score': 1, 
     'Shots on target_Home': 4, 'Shots on target_Away': 4, 'Fouls committed_Home': 15, 'Fouls committed_Away': 13, 
     'Corners_Home': 7, 'Corners_Away': 4, 'Throws_Home': 23, 'Throws_Away': 16, 'Offsides_Home': 5, 'Offsides_Away': 1, 
     'Tackles won_Home': '7 (41%)', 'Tackles won_Away': '13 (62%)', 'Keeper saves_Home': 3, 'Keeper saves_Away': 4, 
     'Yellow cards_Home': 1, 'Yellow cards_Away': 0, 'Red cards_Home': 0, 'Red cards_Away': 0},
    {'Match': 'Cagliari vs Lazio', 'Home Team': 'Lazio', 'Away Team': 'Cagliari', 'Home Score': 2, 'Away Score': 1, 
     'Shots on target_Home': 6, 'Shots on target_Away': 5, 'Fouls committed_Home': 9, 'Fouls committed_Away': 14, 
     'Corners_Home': 7, 'Corners_Away': 7, 'Throws_Home': 11, 'Throws_Away': 19, 'Offsides_Home': 2, 'Offsides_Away': 1, 
     'Tackles won_Home': '7 (44%)', 'Tackles won_Away': '7 (54%)', 'Keeper saves_Home': 4, 'Keeper saves_Away': 4, 
     'Yellow cards_Home': 0, 'Yellow cards_Away': 0, 'Red cards_Home': 0, 'Red cards_Away': 0},
    {'Match': 'Lazio vs Monza', 'Home Team': 'Lazio', 'Away Team': 'Monza', 'Home Score': 5, 'Away Score': 1, 
     'Shots on target_Home': 10, 'Shots on target_Away': 1, 'Fouls committed_Home': 10, 'Fouls committed_Away': 12, 
     'Corners_Home': 8, 'Corners_Away': 1, 'Throws_Home': 10, 'Throws_Away': 8, 'Offsides_Home': 2, 'Offsides_Away': 1, 
     'Tackles won_Home': '8 (80%)', 'Tackles won_Away': '4 (40%)', 'Keeper saves_Home': 0, 'Keeper saves_Away': 4, 
     'Yellow cards_Home': 0, 'Yellow cards_Away': 1, 'Red cards_Home': 0, 'Red cards_Away': 0},
    {'Match': 'Lazio vs Napoli', 'Home Team': 'Lazio', 'Away Team': 'Napoli', 'Home Score': 2, 'Away Score': 2, 
     'Shots on target_Home': 5, 'Shots on target_Away': 2, 'Fouls committed_Home': 9, 'Fouls committed_Away': 19, 
     'Corners_Home': 7, 'Corners_Away': 1, 'Throws_Home': 15, 'Throws_Away': 15, 'Offsides_Home': 1, 'Offsides_Away': 1, 
     'Tackles won_Home': '6 (86%)', 'Tackles won_Away': '10 (56%)', 'Keeper saves_Home': 1, 'Keeper saves_Away': 3, 
     'Yellow cards_Home': 2, 'Yellow cards_Away': 2, 'Red cards_Home': 0, 'Red cards_Away': 0},
    {'Match': 'Venezia vs Lazio', 'Home Team': 'Lazio', 'Away Team': 'Venezia', 'Home Score': 0, 'Away Score': 0, 
     'Shots on target_Home': 3, 'Shots on target_Away': 2, 'Fouls committed_Home': 12, 'Fouls committed_Away': 21, 
     'Corners_Home': 6, 'Corners_Away': 6, 'Throws_Home': 22, 'Throws_Away': 16, 'Offsides_Home': 1, 'Offsides_Away': 0, 
     'Tackles won_Home': '8 (73%)', 'Tackles won_Away': '14 (58%)', 'Keeper saves_Home': 2, 'Keeper saves_Away': 3, 
     'Yellow cards_Home': 2, 'Yellow cards_Away': 4, 'Red cards_Home': 0, 'Red cards_Away': 0}
]

df = pd.DataFrame(data)

# Convert percentage values in 'Tackles won' columns to numerical values
def extract_percentage(value):
    return int(re.search(r'\d+', value).group())

df['Tackles won_Home'] = df['Tackles won_Home'].apply(extract_percentage)
df['Tackles won_Away'] = df['Tackles won_Away'].apply(extract_percentage)

# Add a column to indicate if the team is home or away
df['is_home_team1'] = df['Home Team'] == df['team1']
df['is_home_team2'] = df['Away Team'] == df['team2']

# Drop unnecessary columns
df.drop(columns=['Match'], inplace=True)

# Handle missing values
df.fillna(0, inplace=True)

# Create new features
df['goal_difference'] = df['Home Score'] - df['Away Score']
df['foul_difference'] = df['Fouls committed_Home'] - df['Fouls committed_Away']
df['possession_difference'] = df['possession_team1'] - df['possession_team2']

# Encode categorical variables
categorical_features = ['Home Team', 'Away Team']
numeric_features = [col for col in df.columns if col not in categorical_features + ['result']]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X = df.drop(columns=['result'])
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##############################################################################################

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

# Create a pipeline for classification (win/lose/draw)
pipeline_classification = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', RandomForestClassifier())])

# Create a pipeline for regression (goals, corners, fouls, offside, possession)
pipeline_regression = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor())])

# Train the classification model
pipeline_classification.fit(X_train, y_train)

# Train the regression models
y_goals_home = df['Home Score']
y_goals_away = df['Away Score']
y_corners_home = df['Corners_Home']
y_corners_away = df['Corners_Away']
y_fouls_home = df['Fouls committed_Home']
y_fouls_away = df['Fouls committed_Away']
y_offsides_home = df['Offsides_Home']
y_offsides_away = df['Offsides_Away']
y_keeper_saves_home = df['Keeper saves_Home']
y_keeper_saves_away = df['Keeper saves_Away']
y_tackles_won_home = df['Tackles won_Home']
y_tackles_won_away = df['Tackles won_Away']
y_yellow_cards_home = df['Yellow cards_Home']
y_yellow_cards_away = df['Yellow cards_Away']
y_red_cards_home = df['Red cards_Home']
y_red_cards_away = df['Red cards_Away']

pipeline_goals_home = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor())])
pipeline_goals_away = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor())])
pipeline_corners_home = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', RandomForestRegressor())])
pipeline_corners_away = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', RandomForestRegressor())])
pipeline_fouls_home = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor())])
pipeline_fouls_away = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor())])
pipeline_offsides_home = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor())])
pipeline_offsides_away = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor())])
pipeline_keeper_saves_home = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', RandomForestRegressor())])
pipeline_keeper_saves_away = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', RandomForestRegressor())])
pipeline_tackles_won_home = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', RandomForestRegressor())])
pipeline_tackles_won_away = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', RandomForestRegressor())])
pipeline_yellow_cards_home = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', RandomForestRegressor())])
pipeline_yellow_cards_away = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', RandomForestRegressor())])
pipeline_red_cards_home = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('regressor', RandomForestRegressor())])
pipeline_red_cards_away = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('regressor', RandomForestRegressor())])

pipeline_goals_home.fit(X_train, y_goals_home)
pipeline_goals_away.fit(X_train, y_goals_away)
pipeline_corners_home.fit(X_train, y_corners_home)
pipeline_corners_away.fit(X_train, y_corners_away)
pipeline_fouls_home.fit(X_train, y_fouls_home)
pipeline_fouls_away.fit(X_train, y_fouls_away)
pipeline_offsides_home.fit(X_train, y_offsides_home)
pipeline_offsides_away.fit(X_train, y_offsides_away)
pipeline_keeper_saves_home.fit(X_train, y_keeper_saves_home)
pipeline_keeper_saves_away.fit(X_train, y_keeper_saves_away)
pipeline_tackles_won_home.fit(X_train, y_tackles_won_home)
pipeline_tackles_won_away.fit(X_train, y_tackles_won_away)
pipeline_yellow_cards_home.fit(X_train, y_yellow_cards_home)
pipeline_yellow_cards_away.fit(X_train, y_yellow_cards_away)
pipeline_red_cards_home.fit(X_train, y_red_cards_home)
pipeline_red_cards_away.fit(X_train, y_red_cards_away)

####################################################################################################

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score

# Evaluate the classification model
y_pred_classification = pipeline_classification.predict(X_test)
print("Classification Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classification))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classification))
print("\nClassification Accuracy Score:")
print(accuracy_score(y_test, y_pred_classification))

# Evaluate the regression models
y_pred_goals_home = pipeline_goals_home.predict(X_test)
y_pred_goals_away = pipeline_goals_away.predict(X_test)
y_pred_corners_home = pipeline_corners_home.predict(X_test)
y_pred_corners_away = pipeline_corners_away.predict(X_test)
y_pred_fouls_home = pipeline_fouls_home.predict(X_test)
y_pred_fouls_away = pipeline_fouls_away.predict(X_test)
y_pred_offsides_home = pipeline_offsides_home.predict(X_test)
y_pred_offsides_away = pipeline_offsides_away.predict(X_test)
y_pred_keeper_saves_home = pipeline_keeper_saves_home.predict(X_test)
y_pred_keeper_saves_away = pipeline_keeper_saves_away.predict(X_test)
y_pred_tackles_won_home = pipeline_tackles_won_home.predict(X_test)
y_pred_tackles_won_away = pipeline_tackles_won_away.predict(X_test)
y_pred_yellow_cards_home = pipeline_yellow_cards_home.predict(X_test)
y_pred_yellow_cards_away = pipeline_yellow_cards_away.predict(X_test)
y_pred_red_cards_home = pipeline_red_cards_home.predict(X_test)
y_pred_red_cards_away = pipeline_red_cards_away.predict(X_test)

print("\nRegression Goals Home MAE:")
print(mean_absolute_error(y_test_goals_home, y_pred_goals_home))
print("\nRegression Goals Away MAE:")
print(mean_absolute_error(y_test_goals_away, y_pred_goals_away))
print("\nRegression Corners Home MAE:")
print(mean_absolute_error(y_test_corners_home, y_pred_corners_home))
print("\nRegression Corners Away MAE:")
print(mean_absolute_error(y_test_corners_away, y_pred_corners_away))
print("\nRegression Fouls Home MAE:")
print(mean_absolute_error(y_test_fouls_home, y_pred_fouls_home))
print("\nRegression Fouls Away MAE:")
print(mean_absolute_error(y_test_fouls_away, y_pred_fouls_away))
print("\nRegression Offsides Home MAE:")
print(mean_absolute_error(y_test_offsides_home, y_pred_offsides_home))
print("\nRegression Offsides Away MAE:")
print(mean_absolute_error(y_test_offsides_away, y_pred_offsides_away))
print("\nRegression Keeper Saves Home MAE:")
print(mean_absolute_error(y_test_keeper_saves_home, y_pred_keeper_saves_home))
print("\nRegression Keeper Saves Away MAE:")
print(mean_absolute_error(y_test_keeper_saves_away, y_pred_keeper_saves_away))
print("\nRegression Tackles Won Home MAE:")
print(mean_absolute_error(y_test_tackles_won_home, y_pred_tackles_won_home))
print("\nRegression Tackles Won Away MAE:")
print(mean_absolute_error(y_test_tackles_won_away, y_pred_tackles_won_away))
print("\nRegression Yellow Cards Home MAE:")
print(mean_absolute_error(y_test_yellow_cards_home, y_pred_yellow_cards_home))
print("\nRegression Yellow Cards Away MAE:")
print(mean_absolute_error(y_test_yellow_cards_away, y_pred_yellow_cards_away))
print("\nRegression Red Cards Home MAE:")
print(mean_absolute_error(y_test_red_cards_home, y_pred_red_cards_home))
print("\nRegression Red Cards Away MAE:")
print(mean_absolute_error(y_test_red_cards_away, y_pred_red_cards_away))

####################################################################################

from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search_classification = GridSearchCV(pipeline_classification, param_grid, cv=5, scoring='accuracy')
grid_search_classification.fit(X_train, y_train)

best_params_classification = grid_search_classification.best_params_
best_model_classification = grid_search_classification.best_estimator_

print("Best Parameters for Classification:", best_params_classification)

# Repeat for regression models
param_grid_regression = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search_goals_home = GridSearchCV(pipeline_goals_home, param_grid_regression, cv=5, scoring='neg_mean_absolute_error')
grid_search_goals_home.fit(X_train, y_goals_home)

best_params_goals_home = grid_search_goals_home.best_params_
best_model_goals_home = grid_search_goals_home.best_estimator_

print("Best Parameters for Goals Home:", best_params_goals_home)

# Repeat for other regression models