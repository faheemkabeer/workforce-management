from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the chatbot session
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=(
        "You are a recruitment assistant. Answer questions and assist users with recruitment "
        "and scheduling tasks. Be professional and clear."
    ),
)
chat_session = model.start_chat(history=[])

# Load the UI file
ui, _ = loadUiType("heath.ui")


class MainApp(QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        # Default tab is "Home"
        self.tabWidget.setCurrentIndex(0)

        # Button connections
        self.REC.clicked.connect(self.open_recruitment_tab)
        self.REC_2.clicked.connect(self.open_scheduling_tab)
        self.BACK.clicked.connect(self.back_to_home)
        self.BACK_2.clicked.connect(self.process_recruitment)
        self.BACK_3.clicked.connect(self.back_to_home)
        self.REC_3.clicked.connect(self.start_chatbot)

        # Load the dataset
        try:
            self.dataset = pd.read_csv("watson_healthcare_modified.csv")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Dataset file not found!")
            self.dataset = None

        # Initialize database
        self.create_database()

    def open_recruitment_tab(self):
        """Open Recruitment Tab"""
        self.tabWidget.setCurrentIndex(1)

    def open_scheduling_tab(self):
        """Open Scheduling Tab"""
        self.tabWidget.setCurrentIndex(2)

    def back_to_home(self):
        """Return to Home Tab"""
        self.tabWidget.setCurrentIndex(0)

    def process_recruitment(self):
        """Process recruitment form and validate inputs"""
        name = self.lineEdit.text()
        age = self.lineEdit_2.text()
        experience = self.lineEdit_3.text()

        if not name or not age or not experience:
            QMessageBox.warning(self, "Input Error", "All fields are required!")
            return

        try:
            age = int(age)
            experience = int(experience)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Age and Experience must be valid numbers!")
            return

        if not (22 <= age <= 50):
            QMessageBox.warning(self, "Validation Failed", "Age must be between 22 and 50.")
            return

        if experience < 3:
            QMessageBox.warning(self, "Validation Failed", "Experience must be at least 3 years.")
            return

        if self.dataset is not None:
            matches = self.dataset[(self.dataset["Age"] == age) & (self.dataset["TotalWorkingYears"] >= experience)]
            if matches.empty:
                QMessageBox.warning(self, "No Match", "No matching records found in the dataset.")
                return

        # Save recruitment details to the database
        self.save_to_database(name, age, experience)
        QMessageBox.information(self, "Success", "Recruitment details saved successfully!")

        # Clear input fields
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()

    def create_database(self):
        """Initialize the database for recruitment"""
        conn = sqlite3.connect("recruitment.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recruitment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                experience TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def save_to_database(self, name, age, experience):
        """Save recruitment data into the database"""
        conn = sqlite3.connect("recruitment.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO recruitment (name, age, experience) VALUES (?, ?, ?)", (name, age, experience))
        conn.commit()
        conn.close()

    def start_chatbot(self):
        """Start the chatbot interaction"""
        user_input = self.lineEdit_4.text()
        if not user_input:
            QMessageBox.warning(self, "Input Error", "Please enter a message.")
            return

        try:
            response = chat_session.send_message(user_input)
            bot_response = response.text
            self.lineEdit_4.clear()
            QMessageBox.information(self, "Chatbot Response", f"Bot: {bot_response}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Chatbot error: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()