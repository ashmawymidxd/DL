import requests
import pandas as pd

food_api_url = "http://127.0.0.1:8000/api/patientRatingFood"
response = requests.get(food_api_url)

if response.status_code == 200:
    food_data = response.json()
    food_df = pd.DataFrame(food_data)
    # Save to database or CSV
    food_df.to_csv('C:\\Users\\1\\Desktop\DL\\RecomDietary\\Data\\user_food_interactions.csv', index=False)
else:
    print("Failed to fetch data from API")
