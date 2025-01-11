import streamlit as st
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
# client=OpenAI(api_key='OPENAI_API_KEY')

# Define email mapping for categories, including "Others"
category_emails = {
    "Technical Issues": "techsupport@example.com",
    "Account Issues": "accounts@example.com",
    "Billing & Payment": "billing@example.com",
    "General Inquiry": "info@example.com",
    "Service Requests": "services@example.com",
    "Others": "miscellaneous@example.com",  # Default for unrecognized categories
}

# Define function to classify ticket using GPT
def classify_ticket_gpt(ticket_text):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""
            Classify the following service desk ticket into one of these categories:
            - Technical Issues
            - Account Issues
            - Billing & Payment
            - General Inquiry
            - Service Requests

            If the ticket doesn't match any of these categories, classify it as "Others."

            Ticket Description: {ticket_text}
            Category:
            """
        }
    ],
    temperature=0
    )
    category = response.choices[0]['message']['content'].strip()
    return category if category in category_emails else "Others"

# Define function to send email
def send_email(category, ticket_text):
    sender_email = "your_email@example.com"
    sender_password = "your_email_password"
    recipient_email = category_emails.get(category, category_emails["Others"])  # Default to "Others"
    
    # Create email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"New Ticket - {category}"
    
    body = f"A new ticket has been submitted under the category '{category}'.\n\nTicket Description:\n{ticket_text}"
    msg.attach(MIMEText(body, "plain"))
    
    # Send email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"

# Streamlit UI
st.title("Service Desk Ticket Classifier with Email Routing")
st.write("Enter a service desk ticket below to classify it and route it to the appropriate department.")

# Input text
ticket_text = st.text_area("Ticket Description", "Enter ticket description here...")

# Predict and route on button click
if st.button("Classify and Route Ticket"):
    if ticket_text:
        # Classify using GPT
        category = classify_ticket_gpt(ticket_text)
        
        # Display the category
        st.write(f"Predicted Category: {category}")
        
        # Send email to the appropriate department
        email_status = send_email(category, ticket_text)
        st.write(email_status)
    else:
        st.write("Please enter a ticket description.")