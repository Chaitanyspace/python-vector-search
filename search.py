import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import random

# Initialize the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI setup
st.title("Semantic Search with Manticore")

# Expanded static data to be used in the application
static_data = []

# Generate 1000 items of sample data
descriptions = [
    "Wireless Mouse with ergonomic design and high precision.",
    "Mechanical Keyboard with durable switches and customizable RGB lighting.",
    "USB-C Hub with multiple ports including HDMI, USB 3.0, and Ethernet.",
    "Noise Cancelling Headphones with over-ear design and long battery life.",
    "Portable SSD with high capacity and fast transfer speeds.",
    "Gaming Laptop with NVIDIA graphics, fast processor, and ample RAM.",
    "Adjustable Smartphone Stand for viewing comfort.",
    "Portable Bluetooth Speaker with deep bass and extended battery life.",
    "Ergonomic Office Chair with lumbar support and adjustable features.",
    "High-resolution 4K Monitor with HDR and high refresh rate.",
    "Wireless Earbuds with noise cancellation and touch controls.",
    "Graphics Tablet for digital artists with pressure-sensitive stylus.",
    "Smartwatch with health tracking, GPS, and multiple activity modes.",
    "Portable Projector with HD resolution and built-in speakers.",
    "Standing Desk with height adjustment for ergonomic working.",
    "WiFi Range Extender to boost home WiFi coverage.",
    "Electric Kettle with temperature control and quick boiling.",
    "Air Purifier with HEPA filter for improved indoor air quality.",
    "Fitness Tracker with step counting, sleep monitoring, and heart rate tracking.",
    "Robot Vacuum Cleaner with smart navigation and app control.",
    "Smart Thermostat for remote temperature control.",
    "Electric Toothbrush with multiple modes and extended battery life.",
    "Digital Air Fryer with large capacity and programmable settings.",
    "Programmable Coffee Maker with brew strength control.",
    "External Hard Drive with large storage and fast connectivity.",
    "Noise Reduction Microphone for podcasts and streaming.",
    "Portable Solar Charger for charging devices outdoors.",
    "Portable Air Conditioner with multiple cooling modes.",
    "Wireless Charging Pad for fast, convenient charging.",
    "Smart Doorbell with video recording and two-way audio.",
    "Smart Bulb with customizable colors and remote control.",
    "Pet Feeder with automatic feeding schedule.",
    "Electric Scooter with long-range battery and fast charging.",
    "Bluetooth Tracker for locating keys or bags.",
    "Smart Lock for secure keyless entry.",
    "Digital Camera with 4K video recording capability.",
    "Portable Power Bank with high capacity for multiple devices.",
    "Home Security Camera with night vision and motion detection.",
    "Wireless Game Controller for PC and console gaming.",
    "Smart Light Strip with app control and music sync.",
    "Cordless Vacuum Cleaner with strong suction power.",
    "Drone with HD camera and extended flight time.",
    "Kids Tablet with parental control features.",
    "Video Doorbell with motion detection and instant alerts.",
    "Electric Bike with pedal assist and long battery life.",
    "Waterproof Action Camera for extreme sports.",
    "Portable Bluetooth Keyboard for tablets and smartphones.",
    "Smart Ceiling Fan with remote and app control.",
    "LED Desk Lamp with adjustable brightness and USB charging port."
]

for i in range(1, 1001):
    item = {
        "id": i,
        "name": f"Product {i}",
        "description": random.choice(descriptions)
    }
    static_data.append(item)

# User input for the search query
query = st.text_input("Enter your search query:")

# Button to trigger the search
if st.button("Search"):
    if query:
        # Generate embedding for the query
        query_embedding = model.encode([query])[0]

        # Compute similarity with static data using cosine similarity
        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Generate embeddings for static data descriptions
        static_data_with_embeddings = [
            {**item, "embedding": model.encode([item['description']])[0]} for item in static_data
        ]

        # Calculate similarity between the query and each static item
        results = []
        for item in static_data_with_embeddings:
            similarity = cosine_similarity(query_embedding, item['embedding'])
            results.append({"item": item, "similarity": similarity})

        # Sort results by similarity in descending order
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

        # Display the top 5 results
        st.write("### Search Results:")
        for result in results[:5]:
            st.write(f"**Product Name**: {result['item']['name']}")
            st.write(f"**Description**: {result['item']['description']}")
            st.write(f"**Similarity Score**: {result['similarity']:.4f}")
            st.write("---")
    else:
        st.write("Please enter a query to search.")
