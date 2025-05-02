# SecureScan: Credit Card Fraud Detection Web App

SecureScan is a modern, educational, and production-ready web application for credit card fraud detection. It combines a robust Flask backend with a visually appealing, user-friendly frontend. The platform offers real-time and batch fraud detection, analytics, and a suite of educational resources to help users understand and prevent credit card fraud.

---

## Features

- **Real-Time Fraud Detection:** Instantly check transactions for fraud risk using advanced machine learning.
- **Batch Upload:** Upload CSV files for bulk fraud analysis.
- **Analytics Dashboard:** Visualize fraud trends, rates, and categories.
- **Educational Hub:** Learn about credit scores, smart card usage, fraud types, and FAQs.
- **Modern UI:** Responsive, accessible, and visually appealing frontend.
- **Comprehensive Help & Support:** Built-in FAQ and contact form.

---

## Project Structure

```
Project/
│
├── backend/                # Flask backend (API, model loading, training)
│   ├── app1.py
│   ├── train_model.py
│   ├── requirements.txt
│   └── fraud_detection.db
│
├── frontend/               # All HTML, CSS, and static assets
│   ├── index.html
│   ├── dashboard.html
│   ├── batch.html
│   ├── check.html
│   ├── history.html
│   ├── credit_score.html
│   ├── smart_usage.html
│   ├── fraud_explained.html
│   ├── credit_faq.html
│   ├── help.html
│   ├── styles.css
│   └── sample_batch_upload.csv
│
├── model/                  # Trained ML models and encoders
│   ├── fraud_detection_model.joblib
│   ├── scaler.joblib
│   └── label_encoders.joblib
│
├── data/                   # Datasets and sample data
│   ├── creditcardfraud.csv
│   ├── sample_batch_data.csv
│   ├── sample_fraud_data.csv
│   └── fraud_detection.db
│
└── README.md
```

---

## Setup Instructions

### 1. Backend (Flask API)

**Requirements:** Python 3.8+, pip

1. Navigate to the backend directory:
   ```bash
   cd Project/backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model files are present in `../model/` and the database in `../data/`.

4. Run the Flask app:
   ```bash
   python app1.py
   ```
   The API will be available at **http://localhost:5000/** by default.

---

### 2. Frontend

1. Open any of the HTML files in the `Project/frontend/` directory in your browser (e.g., `index.html`).
2. The frontend communicates with the backend API for predictions and analytics.

---

### 3. Model Training (Optional)

- To retrain the model, use `train_model.py` in the backend directory. Make sure your data is in the `data/` folder.

---

## Key Pages

- **Home:** Overview and quick access to features.
- **Dashboard:** Visual analytics and fraud trends.
- **Check Transaction:** Real-time fraud check for a single transaction.
- **Batch Upload:** Upload CSVs for bulk fraud analysis.
- **History:** View and filter past predictions.
- **Credit Score Tips:** Learn how to maintain a good credit score.
- **Smart Usage:** Best practices for using credit cards wisely.
- **Fraud Explained:** Understand credit card fraud and SecureScan's approach.
- **Credit Card FAQs:** Answers to common credit card and security questions.
- **Help:** FAQ, support contact form, and resources.

---

## Sample Data

- Use `frontend/sample_batch_upload.csv` or `data/sample_batch_data.csv` for testing batch uploads.

---

## Customization

- **Styling:** Edit `frontend/styles.css` for custom themes.
- **Model:** Replace or retrain the model in `model/` as needed.
- **Database:** Use your own data in the `data/` directory.

---



## Ports

- **Backend (Flask API):** Runs on [http://localhost:5000/](http://localhost:5000/) by default.
- **Frontend:** Open HTML files directly in your browser. If you use a local server (e.g., Live Server), use the port it provides (commonly 5500 or 8000).

---

## License

- This project is licensed under the [MIT License](LICENSE).
- See the LICENSE file for details. 