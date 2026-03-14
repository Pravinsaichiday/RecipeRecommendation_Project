import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import re 

# --- 1. LOAD ALL FILES ---
try:
    df = pd.read_csv('recipe_app_data.csv')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
except FileNotFoundError:
    st.error("ERROR: Model files not found. Please run 'build_hackathon_model.py' first.")
    st.stop()
except KeyError:
    st.error("ERROR: Your 'recipe_app_data.csv' is old. Please re-run 'build_hackathon_model.py'.")
    st.stop()

# --- 2. RECOMMENDATION LOGIC ---
def get_recommendations(search_query, top_n=50):
    if not search_query:
        return pd.DataFrame()
    query_tfidf = tfidf.transform([search_query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[:-top_n-1:-1]
    recommended_df = df.iloc[top_indices].copy()
    recommended_df['similarity_score'] = [similarities[i] for i in top_indices]
    return recommended_df

# --- 3. HELPER FUNCTIONS ---
def format_ingredients(ingredients_str):
    """Formats the ingredients string into a neat, simple list for display."""
    ingredients_str = str(ingredients_str)
    ingredients_list = re.split(r'[,\n]+', ingredients_str)
    
    formatted_list = ""
    for item in ingredients_list:
        if item.strip():
            formatted_list += f"{item.strip()}\n"
    return formatted_list

def format_instructions(instructions_str):
    """Formats the instructions string into a neat numbered list for display."""
    instructions_str = str(instructions_str)
    sentences = re.split(r'(?<=\.)\s+', instructions_str) 
    
    if len(sentences) < 2: 
        sentences = instructions_str.split('\n')
        
    markdown_list = ""
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            markdown_list += f"{i+1}. {sentence.strip()}\n"
    return markdown_list

# --- 4. THE STREAMLIT WEB APP ---
st.set_page_config(page_title="FlavorFind AI", layout="wide")
st.title("🧑‍🍳 FlavorFind AI")
st.write("Find your next favorite meal, tailored just for you.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Find Your Perfect Recipe")

st.sidebar.subheader("🌱 Dietary Preferences")
diet_list = ['Any'] + sorted(df['diet'].unique())
selected_diet = st.sidebar.selectbox("Select your diet", diet_list)

st.sidebar.subheader("🌎 Cuisine Type")
cuisine_list = ['Any'] + sorted(df['cuisine'].unique())
selected_cuisine = st.sidebar.selectbox("Select a cuisine", cuisine_list)

st.sidebar.subheader("🔍 Keyword Constraints")
keywords = st.sidebar.text_input(
    "e.g., 'chicken', 'pasta', 'spicy'",
    placeholder="Type keywords..."
)

st.sidebar.subheader("🚫 Allergy Preferences")
allergies_input = st.sidebar.text_input(
    "e.g., 'peanuts, dairy, gluten'",
    placeholder="Comma-separated allergies..."
)

st.sidebar.subheader("⏱️ Max Prep Time")
max_time = st.sidebar.number_input("Maximum time in minutes (optional)", min_value=0, value=0)

st.sidebar.subheader("📊 Number of Recipes")
num_to_show = st.sidebar.slider("Number of Recipes to Show", min_value=1, max_value=20, value=5)

# --- START: NEW FOOTER (UPDATED) ---
st.sidebar.markdown("---") # Adds a horizontal line
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 0.9em;">
        <b>Made by Team - 4 FEDF S-3</b><br>
        C.Pravin Sai - 2420030777<br>
        N. Rithvik Sai - 2420030333<br>
        M.Abrar Hussain - 2420090140
    </div>
    """,
    unsafe_allow_html=True
)
# --- END: NEW FOOTER ---

# --- Main App Logic ---
if st.sidebar.button("Get Recommendations"):
    
    search_query = keywords
    if selected_cuisine != 'Any':
        search_query = selected_cuisine + " " + search_query
    if selected_diet != 'Any':
        search_query = selected_diet + " " + search_query
    
    if not search_query:
        st.warning("Please enter some keywords or select a filter.")
    else:
        recommendations = get_recommendations(search_query, top_n=100)
        final_results = recommendations.copy()
        
        # --- START: NEW FILTERING LOGIC ---
        
        # 1. Filter by Time
        if max_time > 0:
            final_results = final_results[final_results['prep_time_minutes'] <= max_time]
        
        # 2. Filter by Dropdowns
        if selected_cuisine != 'Any':
            final_results = final_results[final_results['cuisine'] == selected_cuisine]
        if selected_diet != 'Any':
            final_results = final_results[final_results['diet'] == selected_diet]
        
        # 3. Filter by Allergies
        allergy_list = [allergy.strip().lower() for allergy in allergies_input.split(',') if allergy.strip()]
        if allergy_list:
            filtered_for_allergies = final_results.copy()
            for allergy in allergy_list:
                rows_to_drop = filtered_for_allergies[
                    filtered_for_allergies['ingredients'].str.lower().str.contains(allergy)
                ].index
                filtered_for_allergies = filtered_for_allergies.drop(rows_to_drop)
            final_results = filtered_for_allergies
            
        # --- END: NEW FILTERING LOGIC ---
            
        final_results = final_results.head(num_to_show)

        # 5. Display the results
        if not final_results.empty:
            st.success(f"Found {len(final_results)} matching recipes!")
            
            for index, row in final_results.iterrows():
                st.subheader(row['name'])
                
                if pd.notna(row['description']):
                    st.markdown(f"*{row['description']}*")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Total Time:** {row['prep_time_minutes']} min")
                with col2:
                    st.markdown(f"**Serves:** 2-4 (est.)")
                
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Ingredients")
                    st.text(format_ingredients(row['ingredients']))
                
                with col2:
                    st.markdown("#### Instructions")
                    st.markdown(format_instructions(row['instructions']))
                
                plain_ingredients = format_ingredients(row['ingredients']).replace('*', '-')
                plain_instructions = format_instructions(row['instructions'])
                
                recipe_text = f"""
{row['name'].upper()}

DESCRIPTION:
{row['description']}

Total Time: {row['prep_time_minutes']} min
Serves: 2-4 (est.)

---

INGREDIENTS:
{plain_ingredients}
---

INSTRUCTIONS:
{plain_instructions}
"""
                file_name = f"{row['name']}.txt".replace(' ', '_').lower()

                st.download_button(
                    label="Save Recipe",
                    data=recipe_text,
                    file_name=file_name,
                    mime="text/plain", 
                )
                
                st.divider() 
        else:
            st.warning("No recipes found matching all your criteria. Try being less specific.")
else:
    st.info("Set your preferences in the sidebar and click 'Get Recommendations'!")
