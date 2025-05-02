from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import os
import logging
from datetime import datetime
import sqlite3
import csv
from io import StringIO

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Load model and encoders
try:
    model = joblib.load('../model/fraud_detection_model.joblib')
    label_encoders = joblib.load('../model/label_encoders.joblib')
    scaler = joblib.load('../model/scaler.joblib')
    logger.info("Model and encoders loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    label_encoders = None
    scaler = None

# Features expected by the model (no merchant_id)
FEATURES = [
    'business_category', 'business_age', 'country',
    'monthly_revenue', 'avg_transaction_value', 'chargeback_ratio',
    'negative_reviews_ratio', 'previous_fraud_reports', 'suspicious_activity_flag'
]
CATEGORICAL = ['business_category', 'country', 'suspicious_activity_flag']

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('../data/fraud_detection.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_id TEXT,
            business_category TEXT,
            business_age INTEGER,
            country TEXT,
            monthly_revenue REAL,
            avg_transaction_value REAL,
            chargeback_ratio REAL,
            negative_reviews_ratio REAL,
            previous_fraud_reports INTEGER,
            suspicious_activity_flag INTEGER,
            prediction INTEGER,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def preprocess_input(data):
    """Preprocess input data for prediction."""
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    categorical_cols = ['business_category', 'country', 'suspicious_activity_flag']
    for col in categorical_cols:
        if label_encoders and col in label_encoders:
            # Handle unseen categories
            if data[col] not in label_encoders[col].classes_:
                logger.warning(f"Unseen value '{data[col]}' for {col}, encoding as 'unknown'")
                df[col] = label_encoders[col].transform(['unknown'])[0]
            else:
                df[col] = label_encoders[col].transform([data[col]])[0]
    
    # Scale numerical features
    numerical_cols = ['business_age', 'monthly_revenue', 'avg_transaction_value', 
                     'chargeback_ratio', 'negative_reviews_ratio', 'previous_fraud_reports']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/check_merchant', methods=['POST'])
def check_merchant():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        # Preprocess input
        df = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Store in database
        conn = sqlite3.connect('../data/fraud_detection.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO transactions (
                merchant_id, business_category, business_age, country,
                monthly_revenue, avg_transaction_value, chargeback_ratio,
                negative_reviews_ratio, previous_fraud_reports,
                suspicious_activity_flag, prediction, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('merchant_id', ''),
            data['business_category'],
            data['business_age'],
            data['country'],
            data['monthly_revenue'],
            data['avg_transaction_value'],
            data['chargeback_ratio'],
            data['negative_reviews_ratio'],
            data['previous_fraud_reports'],
            data['suspicious_activity_flag'],
            int(prediction),
            float(probability)
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Input: {data}")
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Read CSV file
        df = pd.read_csv(file)
        
        # Process each row
        results = []
        conn = sqlite3.connect('../data/fraud_detection.db')
        c = conn.cursor()
        
        for _, row in df.iterrows():
            data = row.to_dict()
            try:
                df_processed = preprocess_input(data)
                prediction = model.predict(df_processed)[0]
                probability = model.predict_proba(df_processed)[0][1]
                # Store in database
                c.execute('''
                    INSERT INTO transactions (
                        merchant_id, business_category, business_age, country,
                        monthly_revenue, avg_transaction_value, chargeback_ratio,
                        negative_reviews_ratio, previous_fraud_reports,
                        suspicious_activity_flag, prediction, probability
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('merchant_id', ''),
                    data['business_category'],
                    data['business_age'],
                    data['country'],
                    data['monthly_revenue'],
                    data['avg_transaction_value'],
                    data['chargeback_ratio'],
                    data['negative_reviews_ratio'],
                    data['previous_fraud_reports'],
                    data['suspicious_activity_flag'],
                    int(prediction),
                    float(probability)
                ))
                results.append({
                    'merchant_id': data.get('merchant_id', ''),
                    'business_category': data.get('business_category', ''),
                    'business_age': data.get('business_age', ''),
                    'country': data.get('country', ''),
                    'monthly_revenue': data.get('monthly_revenue', ''),
                    'avg_transaction_value': data.get('avg_transaction_value', ''),
                    'chargeback_ratio': data.get('chargeback_ratio', ''),
                    'negative_reviews_ratio': data.get('negative_reviews_ratio', ''),
                    'previous_fraud_reports': data.get('previous_fraud_reports', ''),
                    'suspicious_activity_flag': data.get('suspicious_activity_flag', ''),
                    'prediction': int(prediction),
                    'probability': float(probability)
                })
            except Exception as row_e:
                logger.error(f"Error processing row: {data} | Error: {str(row_e)}")
                results.append({
                    **data,
                    'error': str(row_e)
                })
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'results': results,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error processing batch upload: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        conn = sqlite3.connect('../data/fraud_detection.db')
        c = conn.cursor()
        
        # Get total count
        c.execute('SELECT COUNT(*) FROM transactions')
        total = c.fetchone()[0]
        
        # Get paginated results
        offset = (page - 1) * per_page
        c.execute('''
            SELECT * FROM transactions 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        ''', (per_page, offset))
        
        columns = [description[0] for description in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        
        return jsonify({
            'results': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/dashboard_stats', methods=['GET'])
def get_dashboard_stats():
    try:
        conn = sqlite3.connect('../data/fraud_detection.db')
        c = conn.cursor()
        
        # Get total transactions
        c.execute('SELECT COUNT(*) FROM transactions')
        total_transactions = c.fetchone()[0]
        
        # Get fraud rate
        c.execute('SELECT COUNT(*) FROM transactions WHERE prediction = 1')
        fraud_count = c.fetchone()[0]
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        
        # Get fraud by category
        c.execute('''
            SELECT business_category, COUNT(*) as count 
            FROM transactions 
            WHERE prediction = 1 
            GROUP BY business_category
        ''')
        fraud_by_category = dict(c.fetchall())
        
        # Get fraud by country
        c.execute('''
            SELECT country, COUNT(*) as count 
            FROM transactions 
            WHERE prediction = 1 
            GROUP BY country
        ''')
        fraud_by_country = dict(c.fetchall())
        
        conn.close()
        
        return jsonify({
            'total_transactions': total_transactions,
            'fraud_count': fraud_count,
            'fraud_rate': fraud_rate,
            'fraud_by_category': fraud_by_category,
            'fraud_by_country': fraud_by_country,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
