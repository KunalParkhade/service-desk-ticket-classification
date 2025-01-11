# 🛠️Service Desk Ticket Classifier

The **Service Desk Ticket Classifier** is an intelligent system that uses a **Convolutional Neural Network (CNN)** to classify service desk tickets into predefined categories and automatically route them to the appropriate departments via email. This streamlines customer service processes, improves response time, and enhances operational efficiency.

## Features 🌟

- **Ticket Classification**: 🏷️Categorizes service desk tickets into predefined categories such as Technical Issues, Account Issues, Billing & Payments, General Inquiries, and Service Requests.
- **Automatic Email Routing**: 📧Routes tickets to the concerned department emails automatically.
- **Interactive User Interface**: 👩‍💻Provides a user-friendly interface built with Streamlit, featuring multiple tabs for seamless navigation.
- **Customizable Categories**: 🔧Easily modify categories to suit your organization’s needs.

## Project Structure 📂

```plaintext
service-desk-ticket-classification/
├── app.py                # Main application file (Streamlit UI)
├── model.py              # CNN model definition
├── data/
│   ├── words.json        # Word-to-index mapping
│   ├── text.json
│   ├── labels.npy        # Predefined ticket labels
│   ├── testing.jpg       # Banner image for UI
├── saved_models/
│   └── model.pth         # Trained model weights
├── requirements.txt      # Python dependencies
├── train.py
├── .gitignore
├── README.md             # Project documentation
```

## Installation 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/KunalParkhade/service-desk-ticket-classification.git
   cd service-desk-ticket-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pretrained model weights and place them in the `saved_models/` directory.

4. Run the application:
   ```bash
   streamlit run app2.py
   ```

## How to Use 📝

### 1. Ticket Classification 🎟️
- Navigate to the **Home** tab.
- Enter the ticket description in the text area.
- Click on the "Classify Ticket" button to get the predicted category.

### 2. Email Routing 📤
- Ensure email configurations (SMTP server, credentials, etc.).
- On ticket classification, the system automatically sends the ticket to the relevant department email.

### 3. Explore Tabs 🌍
- **Home**: 🏠Homepage
- **Ticket Classification**: 📝Classify tickets and view results.
- **About the Model**: 🔍Model details
- **Contact Us**: 📞Contact details of the company given

## Categories 📑
The predefined ticket categories are:

1. **Technical Issues🖥️**: Issues related to hardware, software, or network.
2. **Account Issues🔑**: Queries or requests related to accounts of users..
3. **Billing & Payment💳**: Issues regarding payments.
4. **General Queries❓**: Any other inquiries or tickets.
5. **Service Requests🛠️**: Any service requests.

## Screenshots 📸
![Home Banner](data/testing.jpg)

## Future Enhancements 🔮
- Support for multilingual ticket classification.
- Integration with ticket management platforms like Jira or ServiceNow. 🔗
- Advanced analytics for ticket trends and response metrics.📊

## Contributing 🤝
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your feature 🌱.
3. Submit a pull request with a clear description of your changes.⬇️

## License 📜
This project is licensed under the [MIT License]().

## Contact 📧
For any queries or suggestions, please contact:
- **Name**: Kunal Parkhade
- **Email**: kunalparkhade@gmail.com

---
**Streamline your service desk operations with the power of AI!** 🤖