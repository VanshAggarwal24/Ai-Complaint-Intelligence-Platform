# AI Complaint Intelligence Platform

An AI-powered complaint classification and intelligent business routing system built using Natural Language Processing (NLP) and Machine Learning.

---

## Overview

This application analyzes unstructured financial customer complaints and automatically:

- Classifies the complaint category  
- Calculates model confidence  
- Assigns the appropriate business department  
- Determines complaint priority (High / Medium / Low)  
- Computes a risk score  
- Generates a structured response template  
- Displays model probability breakdown  

The system simulates a real-world enterprise complaint management workflow.

---

## Problem Statement

Financial institutions receive thousands of customer complaints daily.  
Manual classification and routing leads to:

- Delayed response times  
- Incorrect department assignment  
- Inconsistent prioritization  
- Increased operational workload  

This system automates complaint understanding and routing using machine learning to improve efficiency and decision-making.

---

## Tech Stack

- Python  
- Scikit-learn  
- TF-IDF Vectorization  
- Logistic Regression  
- Streamlit  
- NLTK  
- Matplotlib  
- Joblib  

---

## How It Works

1. User enters complaint text  
2. Text is cleaned using NLP preprocessing:
   - Lowercasing  
   - Removing special characters  
   - Stopword removal  
3. TF-IDF converts text into numerical features  
4. Logistic Regression predicts complaint category  
5. System calculates:
   - Model confidence  
   - Priority level  
   - Risk score  
6. Business routing logic assigns department  
7. Response template is automatically generated  

---

## Model Information

- Algorithm: Logistic Regression  
- Feature Extraction: TF-IDF  
- Multi-class classification  
- Accuracy: ~82%  
- Confidence-based risk scoring  
- Manual review fallback for low-confidence predictions  

---

## Run Locally

Clone the repository:

git clone https://github.com/VanshAggarwal24/Ai-Complaint-Intelligence-Platform.git
cd Ai-Complaint-Intelligence-Platform

Install dependencies:

pip install -r requirements.txt  

Run the application:

streamlit run app.py  

---

## Deployment

Deployment Link: https://ai-complaint-intelligence-platform-vansh-aggarwal.streamlit.app/

---

## Business Use Case

This system can be integrated into:

- Banking customer support platforms  
- FinTech complaint portals  
- CRM systems  
- Insurance complaint routing workflows  

It reduces manual effort and improves complaint handling efficiency.

---

## Limitations

- Performance decreases for very short or random inputs  
- Unseen vocabulary may reduce confidence  
- Requires periodic retraining for new complaint categories  

---

## Author

Vansh Aggarwal  
B.Tech Computer Science  
