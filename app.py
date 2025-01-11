import streamlit as st
import torch
from model import TicketClassifier
import numpy as np
import json
import torch.nn.functional as F

# Load word to index dictionary
with open('data/words.json', 'r') as f:
    words = json.load(f)
word2idx = {o: i for i, o in enumerate(words)}

# Define category names (you can modify this list based on your categories)
category_names = ['Technical Issues', 'Account Issues', 'Billing & Payment', 'General Inquiry', 'Service Requests']

# Load model
model = TicketClassifier(len(word2idx) + 1, 64, len(category_names))  # Adjusted for number of categories
model.load_state_dict(torch.load('saved_models/model.pth', weights_only=True))
model.eval()

# Define function for padding and transforming the input text
def pad_input(sentence, seq_len=50):
    sentence = [word2idx.get(word, 0) for word in sentence.split()]
    features = np.zeros((seq_len,), dtype=int)
    features[-len(sentence):] = np.array(sentence)[:seq_len]
    return features

# Streamlit UI elements
st.title("Service Desk Ticket Classifier")
st.write("Enter a service desk ticket below to classify it.")

# Input text
ticket_text = st.text_area("Ticket Description", "Enter ticket description here...")

# Predict on button click
if st.button("Classify Ticket"):
    if ticket_text:
        # Prepare the text input
        input_data = pad_input(ticket_text)
        input_tensor = torch.from_numpy(input_data).unsqueeze(0)  # Add batch dimension

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=-1).item()

        # Display result using the category names
        st.write(f"Predicted Category: {category_names[predicted_class]}")
    else:
        st.write("Please enter a ticket description.")
