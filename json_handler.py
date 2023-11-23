import json
import os

# Folder path to store JSON files
folder_path = "known_faces/data"
os.makedirs(folder_path, exist_ok=True)

# Function to save data to a JSON file
def save_data_to_json(data):
    filename = os.path.join(folder_path, "user_data.json")
    with open(filename, 'w') as file:
        json.dump(data, file)
    print("Data saved to", filename)

# Function to load data from a JSON file
def load_data_from_json():
    filename = os.path.join(folder_path, "user_data.json")
    with open(filename, 'r') as file:
        data = json.load(file)
        return data

# Define the data structure
data = {
    "user_encodings": [],
    "user_names": []
}

save_data_to_json(data)
loaded_data = load_data_from_json()
