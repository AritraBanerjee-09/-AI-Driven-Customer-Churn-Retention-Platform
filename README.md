
# Chubb Churns: Chubb loyalty navigator 

Chubb Sentinel is a full-stack machine learning application built as a proof-of-concept for Chubb. It serves as an end-to-end retention platform that predicts which auto insurance customers are at risk of churning, explains the reasons behind the risk, and prescribes AI-powered actions to help retain them.

### Setup and Installation
Follow these steps to run the project locally.

1.**Folder Structure**

Ensure your project is structured with the archive folder (containing the CSVs) as a sibling to the chubb-sentinel project folder.

```
/your-project-directory
├── /archive
│   └── autoinsurance_churn.csv
└── /chubb-sentinel
    ├── artifacts/
    ├── data/
    ├── pages/
    ├── .streamlit/
    ├── app.py
    └── ...

```
2. **Install Tesseract OCR**
   
This is a required system-level dependency for the "Quote Analyzer" feature.

Debian/Ubuntu/Pop!_OS:
` sudo apt-get update
sudo apt-get install tesseract-ocr`
macOS:
`brew install tesseract`
Windows:
Download and run the installer from the Tesseract at UB Mannheim page.


3. **Create Python Environment & Install Packages**
Navigate into the chubb-sentinel directory and run the following commands:

```
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required Python packages
pip install -r requirements.txt
```


4. **Configure API Key**
The Generative AI features require a Google Gemini API key.

Create a folder named .streamlit inside your chubb-sentinel project.

Inside .streamlit, create a file named secrets.toml.

Add your API key to this file:

```GEMINI_API_KEY = "YOUR_API_KEY_HERE"```

### Usage / Workflow
To run the application, you must first generate the model artifacts.

1. **Generate Model Artifacts**
Run the backend scripts from your terminal in the chubb-sentinel directory.

```
# First, run the data preparation script
python data_prep.py

# Second, run the model training script
python model_trainer.py
```

This will populate the artifacts/ and data/ directories with the necessary files.

2. **Launch the Streamlit App**
`streamlit run app.py`


### Key Features

**Churn Prediction**: Utilizes an XGBoost model trained on a large dataset to generate a precise churn probability score for every customer.

**Explainable AI (XAI) Insights**: Integrates the SHAP library to move beyond a "black box" model, providing:

**Global Insights**: A portfolio-level view of the most important factors driving churn overall.

**Local Explanations**: Per-customer waterfall plots that show the exact reasons for an individual's specific churn score.

**AI-Powered Recommendations**: Employs a hybrid AI approach for actionable insights:

**XAI-Driven Engine**: A rule-based system that translates SHAP outputs into transparent, justifiable retention strategies.

**Generative AI Plans (Google Gemini)**: Creates sophisticated, context-aware action plans and drafts personalized, empathetic emails for agents.

**Multi-Page Interactive Dashboard (Streamlit)**: A comprehensive UI for business users, including:

**Portfolio Dashboard**: KPIs and a geospatial map of at-risk customers.

**Triage List**: A prioritized, actionable list of clients to contact.

**Deep Dive & Action Plan**: A 360-degree view of any at-risk customer.

**Advanced Simulators**: A "Live Simulator" to test hypothetical customer profiles and a "Life Event Monitor" for proactive outreach.

**Conversational AI**: A chatbot for guided, interactive analysis.


### Innovative Tools:

**AI Quote Analyzer**: Uses OCR (Tesseract) and the Gemini API to analyze competitor quotes.

**ROI Calculator**: Quantifies the financial impact of retention efforts.

#### Architecture Overview

The project is structured into a modular pipeline and a user-facing application:

Offline Data Pipeline (data_prep.py): Reads raw data, performs extensive feature engineering (geospatial, household, financial ratios), and saves clean data splits and a preprocessing artifact.

Offline Model Training (model_trainer.py): Loads the prepared data, handles class imbalance with SMOTE, trains the XGBoost model, creates a SHAP explainer, and saves the final model artifacts.

Online Application (app.py & pages/): A multi-page Streamlit application that loads the pre-trained artifacts. It performs live feature engineering and prediction at startup (using caching for performance) and provides the interactive UI for analysis.







