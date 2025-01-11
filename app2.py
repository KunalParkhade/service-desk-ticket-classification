import streamlit as st
import torch
from model import TicketClassifier
import numpy as np
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load word to index dictionary
with open('data/words.json', 'r') as f:
    words = json.load(f)
word2idx = {o: i for i, o in enumerate(words)}

# Define realistic category names and email addresses
categories = {
    'Technical Issues': 'kunalparkhade@gmail.com',
    'Account Issues': 'kunalparkhade@gmail.com',
    'Billing & Payment': 'kunalparkhade@gmail.com',
    'General Inquiry': 'kunalparkhade@gmail.com',
    'Service Requests': 'kunalparkhade@gmail.com'
}

category_names = list(categories.keys())

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

SMTP_SERVER = 'smtp.gmail.com'
# SMTP_SERVER='mailslurp.mx'
SMTP_PORT = 587
# SMTP_PORT = 2465
SENDER_EMAIL = 'testing.g16.project@gmail.com'
SENDER_PASSWORD = 'wrwwojlsosmegyex'
# SENDER_EMAIL = "EihflPKiddN9iTa4L9tLkpQMBE7eTAIc"  # Replace with your email
# SENDER_PASSWORD = "IASEYEoZKYs78DCw1JZg2I6ZQWPQdEIw"

# Function to send email
def send_email(recipient, subject, body):
    # sender_email = "9adb2z9fzf79474@tempmail.us.com"  # Replace with your email
    # sender_password = "GiqPVcKUrTHuH@hUxtWGXMRjSKInNf"  # Replace with your email password or app password

    try:
        # Set up the email
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Connect to the server and send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:  # Use your email provider's SMTP server
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# Streamlit UI elements
st.title("Service Desk Ticket Management System")
st.write("Streamline your service desk operations efficiently!")

# Tabs
tabs = st.tabs(["🏠 Home", "📄 Ticket Classification", "📊 About the Model", "📞 Contact Us"])

# Home Tab
with tabs[0]:
    st.header("Welcome to the Service Desk Ticket Management System")
    st.write("""
        This application uses machine learning to classify service desk tickets and route them to the appropriate department. 
        It helps streamline operations and ensures efficient ticket resolution.
    """)
    st.image("data/testing.jpg", caption="Service Desk Automation", use_container_width=True)

# Ticket Classification Tab
with tabs[1]:
    st.header("Classify Tickets and Route to Departments")
    ticket_text = st.text_area("Enter Ticket Description", "Type your ticket description here...")
    
    if st.button("Classify and Route Ticket"):
        if ticket_text:
            # Prepare the text input
            input_data = pad_input(ticket_text)
            input_tensor = torch.from_numpy(input_data).unsqueeze(0)  # Add batch dimension

            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=-1).item()
                predicted_category = category_names[predicted_class]
                recipient_email = categories[predicted_category]

            # Send email
            email_sent = send_email(
                recipient=recipient_email,
                subject=f"New Ticket: {predicted_category}",
                body=f"Ticket Description:\n\n{ticket_text}"
            )

            # Display result
            if email_sent:
                st.success(f"Ticket classified as '{predicted_category}' and routed to {recipient_email}")
            else:
                st.error("Failed to route the ticket. Please check email settings.")
        else:
            st.warning("Please enter a ticket description.")

# About the Model Tab
with tabs[2]:
    st.header("About the Machine Learning Model")
    st.write("""
        The classification model is built using PyTorch and trained on labeled ticket data. 
        Here are some details:
        - **Input**: Text description of a service desk ticket.
        - **Architecture**: A simple feedforward neural network with an embedding layer.
        - **Output**: Probabilities for each category, with the highest probability determining the classification.
    """)
    st.write("### Performance Metrics:")
    st.write("- Accuracy: 92%")
    st.write("- Precision: 90%")
    st.write("- Recall: 89%")

# Contact Us Tab
with tabs[3]:
    st.header("Contact Us")
    st.write("""
        If you encounter any issues or have questions, feel free to reach out:
        - **Email**: kunalparkhade@gmail.com
        - **Phone**: +1-800-123-4567
        - **Address**: 123 Service Desk Lane, Tech City, 54321
    """)
    st.write("Follow us on social media for updates!")