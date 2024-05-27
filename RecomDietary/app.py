import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from flask import Flask, request, jsonify

# Example API endpoints
user_api = "http://127.0.0.1:8000/api/patientdata"
food_api = "http://127.0.0.1:8000/api/food"

# Fech Patient interaction data and save it in a CSV file
food_api_url = "http://127.0.0.1:8000/api/patientRatingFood"
response = requests.get(food_api_url)

if response.status_code == 200:
    food_data = response.json()
    food_df = pd.DataFrame(food_data)
    # Save to database or CSV
    food_df.to_csv('C:\\Users\\1\\Desktop\DL\\RecomDietary\\user_food_interactions.csv', index=False)
else:
    print("Failed to fetch data from API")

# Fetch user data from API
users_response = requests.get(user_api)
users_json = users_response.json()

if users_response.status_code == 200 and users_json['status']:
    users = pd.DataFrame(users_json['data'])
    # print(users)
else:
    print("Failed to fetch user data")

# Load food data from an API
foods_response = requests.get(food_api)
foods_json = foods_response.json()

if foods_response.status_code == 200 and isinstance(foods_json, list):
    foods = pd.DataFrame(foods_json)
    # print(foods)
else:
    print("Failed to fetch food data")

# Debug: Print columns of foods DataFrame
# print("Columns in foods DataFrame:", foods.columns)

# Load interaction data from a CSV file
interactions = pd.read_csv('C:\\Users\\1\\Desktop\\DL\\RecomDietary\\user_food_interactions.csv')

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
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[user_input, food_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([user_features_np, food_features_np], liked_np, epochs=10, batch_size=32)

# Flask app to serve recommendations
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
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
