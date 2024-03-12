from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load and preprocess the dataset
recipes_df = pd.read_csv("recipes.csv")
recipes_data = recipes_df[['RecipeId', 'Calories']].copy()

# Cluster the data
kmeans = KMeans(n_clusters=3, random_state=42)
recipes_data['Cluster'] = kmeans.fit_predict(recipes_data[['Calories']])


# Implement KNN model
def predict_calorie_cluster(calorie_needed):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(recipes_data[['Calories']], recipes_data['Cluster'])
    predicted_cluster = knn.predict([[calorie_needed]])
    return predicted_cluster[0]


# Calculate BMI
def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)


# Calculate calorie needed based on user data
def calculate_calories_needed(weight, height, age, bmi, steps_count, diet_choice):
    bmr = 10 * weight + 6.25 * height - 5 * age
    if diet_choice == 'weight_gain':
        calorie_needed = bmr + (steps_count * 0.05)
    elif diet_choice == 'weight_loss':
        calorie_needed = bmr - (steps_count * 0.05)
    else:  # balanced weight
        calorie_needed = bmr
    return calorie_needed


# Recommend food items from the predicted cluster with closest calories
def recommend_food(calorie_needed, predicted_cluster):
    # Select recipes from the predicted cluster
    predicted_cluster_recipes = recipes_df[recipes_data['Cluster'] == predicted_cluster]

    # Filter out recipes without images or with "character(0)" in the Images column
    predicted_cluster_recipes = predicted_cluster_recipes[predicted_cluster_recipes['Images'].apply(lambda x: isinstance(x, str) and not x.startswith("character(0)"))]

    if predicted_cluster_recipes.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no recipes have images

    # Calculate the difference in calories for each recipe
    predicted_cluster_recipes['CalorieDifference'] = abs(predicted_cluster_recipes['Calories'] - calorie_needed)

    # Sort the recipes based on calorie difference and select the top 5 closest ones
    top_5_recipes = predicted_cluster_recipes.nsmallest(5, 'CalorieDifference')

    # Clean and split keywords, recipe instructions, and images
    top_5_recipes['Keywords'] = top_5_recipes['Keywords'].apply(clean_and_split_keywords)
    top_5_recipes['RecipeInstructions'] = top_5_recipes['RecipeInstructions'].apply(clean_and_split_instructions)
    top_5_recipes['Images'] = top_5_recipes['Images'].apply(clean_and_split_images)

    # Return the top 5 recipes with additional information
    return top_5_recipes[
        ['RecipeId', 'Name', 'Description', 'Images', 'RecipeCategory', 'Keywords', 'Calories', 'FatContent',
         'ProteinContent', 'CarbohydrateContent', 'RecipeInstructions']]


def clean_and_split_instructions(instructions):
    cleaned_instructions = instructions.replace('c("', '').replace('")', '')
    return cleaned_instructions.split('", "')


def clean_and_split_keywords(keywords):
    if isinstance(keywords, str):
        cleaned_keywords = keywords.replace('c("', '').replace('")', '')
        return cleaned_keywords.split('", "')
    else:
        return []


def clean_and_split_images(images):
    if isinstance(images, str) and images.startswith("c("):
        # Extract URLs from the string within "c()"
        urls = images[3:-2].split('", "')
        # Ensure all URLs have the correct format (e.g., start with "https://")
        urls = [url if url.startswith("https://") else "https://" + url for url in urls]
        return urls[:1]  # Return only the first URL
    else:
        return []





@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    age = int(request.form['age'])
    steps_count = int(request.form['steps_count'])
    diet_choice = request.form['diet_choice'].strip().lower()

    bmi = calculate_bmi(height, weight)
    calories_needed = calculate_calories_needed(weight, height, age, bmi, steps_count, diet_choice)
    predicted_cluster = predict_calorie_cluster(calories_needed)
    recommended_food = recommend_food(calories_needed, predicted_cluster)

    return render_template('result.html', recommended_food=recommended_food)


if __name__ == '__main__':
    app.run(debug=True)
