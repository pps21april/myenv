import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# df = pd.read_csv('dataset.csv')
import random

# Load the dataset
file_path = 'dataset.csv'
sample_size = 5000  # Adjust the sample size as needed

# Set a random seed for reproducibility
random.seed(42)

# Read a random sample of rows from the dataset
df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and random.random() > (sample_size / 30000),on_bad_lines='skip')

# Filling missing values for Hotel Details with 'Not Available'
df['Hotel Details'].fillna('Not Available', inplace=True)

# Filling missing values for Airline with 'Not Available'
df['Airline'].fillna('Not Available', inplace=True)

# Filling missing values for Onwards Return Flight Time with 'Not Available'
df['Onwards Return Flight Time'].fillna('Not Available', inplace=True)

# Filling missing values for Sightseeing Places Covered with 'Not Available'
df['Sightseeing Places Covered'].fillna('Not Available', inplace=True)

# Filling missing values for Initial Payment For Booking with 0 (assuming no initial payment)
df['Initial Payment For Booking'].fillna(0, inplace=True)

# Filling missing values for Cancellation Rules with 'Not Available'
df['Cancellation Rules'].fillna('Not Available', inplace=True)

# Dropping columns with all missing values (Flight Stops, Date Change Rules, Unnamed: 22, Unnamed: 23)
df.drop(columns=["Flight Stops", "Meals", "Initial Payment For Booking", "Date Change Rules"], inplace=True)
df['Travel Date'] = pd.to_datetime(df['Travel Date'], format='%d-%m-%Y', errors='coerce')
allowed_package_types = ['Deluxe', 'Standard', 'Premium', 'Luxury', 'Budget']

# Filter the DataFrame to keep only the rows with allowed package types
df = df[df['Package Type'].isin(allowed_package_types)]
df.drop('Company', axis=1, inplace=True)
df.drop('Crawl Timestamp', axis=1, inplace=True)



# Load the dataset
data = df

# Data Preprocessing
# Combine relevant columns into a single column for hotel information
data['Hotel_Info'] = data['Hotel Details'].str.cat(data['Destination'], sep='|')

# Create a TF-IDF vectorizer to convert text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the vectorizer on the Hotel_Info column
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Hotel_Info'])

# Compute the cosine similarity between hotels based on TF-IDF vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get hotel recommendations based on Package Type, Start City, Price, and Destination
def get_hotel_recommendations(package_type, start_city, price, destination, cosine_sim=cosine_sim):
    # Filter the dataset based on the given criteria
    filtered_data = data[(data['Package Type'] == package_type) &
                         (data['Start City'] == start_city) &
                         (data['Price Per Two Persons'] <= price) &
                         (data['Destination'] == destination)]

    if filtered_data.empty:
        return "No matching hotels found."

    # Get the indices of the filtered hotels
    hotel_indices = filtered_data.index

    # Calculate the average cosine similarity score for each hotel with the filtered hotels
    avg_similarity_scores = []
    for idx in hotel_indices:
        avg_score = sum(cosine_sim[idx]) / len(cosine_sim[idx])
        avg_similarity_scores.append(avg_score)

    # Create a DataFrame to store the filtered hotels and their average similarity scores
    recommended_hotels_df = pd.DataFrame({'Uniq Id': filtered_data['Uniq Id'],
                                          'Hotel Details': filtered_data['Hotel Details'],
                                          'Avg Similarity Score': avg_similarity_scores})

    # Sort the hotels by average similarity score in descending order
    recommended_hotels_df = recommended_hotels_df.sort_values(by='Avg Similarity Score', ascending=False)

    # Return the recommended hotel details
    return recommended_hotels_df[['Uniq Id', 'Hotel Details']]

st.title('Hotel Recommendation App')

# Create dropdowns for Package Type, Start City, and Destination
package_types = df['Package Type'].unique()
start_cities = df['Start City'].unique()
destinations = df['Destination'].unique()

package_type = st.selectbox('Select Package Type:', package_types)
start_city = st.selectbox('Select Start City:', start_cities)
destination = st.selectbox('Select Destination:', destinations, format_func=lambda x: x.replace('|', ', '))

# Slider for Price
price = st.slider('Select Maximum Price:', min_value=0, max_value=df['Price Per Two Persons'].max(), value=10000)

# Get recommendations on button click
if st.button('Get Recommendations'):
    recommended_hotels = get_hotel_recommendations(package_type, start_city, price, destination)
    if isinstance(recommended_hotels, str):
        st.warning(recommended_hotels)
    else:
        st.table(recommended_hotels)

# Optionally, you can display some information about the selected filters
st.write('Selected Filters:')
st.write(f'Package Type: {package_type}')
st.write(f'Start City: {start_city}')
st.write(f'Destination: {destination}')
st.write(f'Maximum Price: {price}')