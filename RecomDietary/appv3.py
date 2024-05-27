import json
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout

# Load Data API endpoints
user_api = "http://127.0.0.1:8000/api/patientdata"
food_api_url = "http://127.0.0.1:8000/api/patientRatingFood"

# Fetch 1)- Patient interaction data and save it in a CSV file
response = requests.get(food_api_url)
if response.status_code == 200:
    food_data = response.json()
    food_df = pd.DataFrame(food_data)
    # Save to database or CSV
    food_df.to_csv('C:\\Users\\1\\Desktop\\DL\\RecomDietary\\Data\\user_food_interactions.csv', index=False)
else:
    print("Failed to fetch data from API")

# Fetch 2)- user data from API
users_response = requests.get(user_api)
users_json = users_response.json()
if users_response.status_code == 200 and users_json['status']:
    users = pd.DataFrame(users_json['data'])
else:
    print("Failed to fetch user data")

# Fetch 3)- food data from JSON file
json_file_path = 'C:\\Users\\1\\Desktop\\DL\\RecomDietary\\Data\\food.json'
try:
    with open(json_file_path, 'r') as file:
        foods_json = json.load(file)
    
    if isinstance(foods_json, list):
        foods = pd.DataFrame(foods_json)
        print("=====> Food data fetched successfully")
    else:
        print("Invalid JSON format: Expected a list")
except FileNotFoundError:
    print(f"File not found: {json_file_path}")
except json.JSONDecodeError:
    print("Error decoding JSON")

# Load interaction data from a CSV file
interactions = pd.read_csv('C:\\Users\\1\\Desktop\\DL\\RecomDietary\\Data\\user_food_interactions.csv')

# Ensure the necessary columns are present
assert 'user_id' in users.columns, "Missing 'user_id' in users DataFrame"
assert 'user_id' in interactions.columns, "Missing 'user_id' in interactions DataFrame"
assert 'food_id' in interactions.columns, "Missing 'food_id' in interactions DataFrame"

# Merge interaction data with user and food features
interactions = interactions.merge(users, on='user_id')
interactions = interactions.merge(foods, on='food_id')

# One-hot encode categorical features
user_features = pd.get_dummies(interactions[['food_id', 'bmi', 'diabetic_status', 'glucose_level', 'favorite_food']])
food_features = pd.get_dummies(interactions[['user_id', 'food_id', 'breakfast', 'lunch', 'dinner', 'totalCalories', 'carbohydrates', 'proteins', 'fats']])
liked = interactions['liked']

# Convert DataFrames to NumPy arrays for TensorFlow
user_features_np = user_features.values.astype(np.float32)
food_features_np = food_features.values.astype(np.float32)
liked_np = liked.values.astype(np.float32)

# Split data into training and validation sets
user_features_train, user_features_val, food_features_train, food_features_val, liked_train, liked_val = train_test_split(
    user_features_np, food_features_np, liked_np, test_size=0.2, random_state=42)

# Normalize the features
print("User features shape:", user_features_np.shape)
print("Food features shape:", food_features_np.shape)
print("Liked array shape:", liked_np.shape)

# Define the model
user_input = Input(shape=(user_features_np.shape[1],))
food_input = Input(shape=(food_features_np.shape[1],))

# Combine features
combined_features = Concatenate()([user_input, food_input])

# Build a neural network
x = Dense(128, activation='relu')(combined_features)
x = Dropout(0.3)(x)  # Add dropout for regularization
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)  # Add dropout for regularization
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[user_input, food_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [user_features_train, food_features_train], liked_train,
    validation_data=([user_features_val, food_features_val], liked_val),
    epochs=10, batch_size=32
)

# Print model accuracy
val_loss, val_accuracy = model.evaluate([user_features_val, food_features_val], liked_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Flask app to serve recommendations
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Fetch user data from api request body
    user_data = request.json['user']
    user_df = pd.DataFrame([user_data])
    user_features_enc = pd.get_dummies(user_df).reindex(columns=user_features.columns, fill_value=0)
    user_features_np = user_features_enc.values.astype(np.float32)
    
    recommendations = []
    for _, food in foods.iterrows():
        food_features_enc = pd.get_dummies(pd.DataFrame([food])).reindex(columns=food_features.columns, fill_value=0)
        food_features_np = food_features_enc.values.astype(np.float32)
        score = model.predict([user_features_np, food_features_np])
        recommendations.append((food['food_id'], float(score[0][0])))  # Convert float32 to float
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)