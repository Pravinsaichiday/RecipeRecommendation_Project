import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import sys
import re # <-- Import the regex library

# --- CONFIGURATION ---
DATA_FILE = 'cuisine_updated.csv' 
print(f"--- HACKATHON MODEL BUILDER (5k Dataset) ---")

# --- NEW: Smarter Cleaning Functions ---

def clean_text_fields(text):
    """This function cleans text for paragraphs (name, description, instructions)"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML
    text = text.replace('\n', ' ') # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()

def clean_list_fields(text):
    """This function cleans text for lists (ingredients)"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text) # ONLY remove HTML
    return text.strip() # Keep newlines

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded '{DATA_FILE}'. Found {len(df)} recipes.")
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{DATA_FILE}' was not found.")
    sys.exit()

# --- 2. PREPARE DATA ---
df_clean = df.dropna(subset=['name', 'ingredients', 'diet', 'cuisine', 'instructions', 'prep_time'])
df_model = df_clean.copy()

def parse_prep_time(time_str):
    # (Prep time function remains the same)
    if pd.isna(time_str): return 0
    time_str = str(time_str).lower()
    total_minutes = 0
    if 'h' in time_str:
        try:
            hours = time_str.split('h')[0].strip()
            total_minutes += int(''.join(filter(str.isdigit, hours))) * 60
        except: pass
    if 'm' in time_str:
        try:
            parts = time_str.split('h')[-1] 
            minutes = parts.split('m')[0].strip()
            total_minutes += int(''.join(filter(str.isdigit, minutes)))
        except: pass
    if 'h' not in time_str and 'm' not in time_str and time_str.isdigit():
        total_minutes = int(time_str)
    return total_minutes

df_model['prep_time_minutes'] = df_model['prep_time'].apply(parse_prep_time)

# --- APPLY THE NEW CLEANING FUNCTIONS ---
print("Cleaning text data (removing HTML, preserving lists)...")
df_model['ingredients'] = df_model['ingredients'].apply(clean_list_fields) # Use new function
df_model['instructions'] = df_model['instructions'].apply(clean_text_fields) # Use old function
df_model['description'] = df_model['description'].apply(clean_text_fields) # Use old function
df_model['name'] = df_model['name'].apply(clean_text_fields) # Use old function

print(f"Cleaned data: {len(df_model)} usable recipes remain.")

# Create the 'features' column
df_model['features'] = (
    df_model['name'] + ' ' + 
    df_model['ingredients'].replace('\n', ' ') + # Smash ingredients for search, but not for saving
    df_model['diet'] + ' ' + 
    df_model['cuisine'] + ' ' +
    df_model['instructions']
)
df_model['features'] = df_model['features'].fillna('')
print("Created combined 'features' column for model.")

# --- 3. BUILD MODEL ---
tfidf = TfidfVectorizer(stop_words='english')
print("Applying TF-IDF vectorization... (This is fast)")
tfidf_matrix = tfidf.fit_transform(df_model['features'])

print("Saving model files...")
# --- 4. SAVE FILES ---
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')

# Save the app data (now with clean, list-formatted text)
app_data = df_model[['name', 'ingredients', 'instructions', 'diet', 'cuisine', 'prep_time_minutes', 'description']]
app_data.to_csv('recipe_app_data.csv', index=False)

print("\n--- MODEL CREATED SUCCESSFULLY! (with better cleaning) ---")
print("Your hackathon model is ready to go.")
