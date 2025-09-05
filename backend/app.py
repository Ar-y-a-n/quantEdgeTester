import re  # ADD THIS IMPORT
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from utils import create_user, authenticate_user, logout_user, is_valid_email
import pandas as pd
import os
import json
import plotly
import numpy as np
from email.mime.text import MIMEText
from email_utils import send_email
import smtplib
import plotly.graph_objects as go
import logging
import csv
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
CSV_FILE = 'strategies.csv'

# Configure logging through Flask app
app.logger.setLevel(logging.INFO)

# Configuration - centralized paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_PATH = os.path.join(DATA_DIR, "frontend_data")
PRELOADED_DATA = {}

# Ensure frontend data directory exists
os.makedirs(FRONTEND_PATH, exist_ok=True)

# Load environment variables for sensitive data
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'mansi24ecell@gmail.com')  # Add to .env


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return "✅ Flask backend is running."
    return jsonify({'message': 'Backend running'}), 200


@app.route('/login', methods=['POST', 'OPTIONS'])  # Support CORS preflight
def login():
    if request.method == 'OPTIONS':
        return '', 200  # CORS preflight OK

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No login data provided'}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '').strip()

    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password required'}), 400

    success, message, user = authenticate_user(email, password)

    if not success:
        return jsonify({'success': False, 'message': message}), 401  # Unauthorized

    return jsonify({
        'success': True,
        'message': message,
        'user': {
            'name': user.get('name'),
            'email': user.get('email')
        }
    }), 200


@app.route('/contact', methods=['POST'])
def contact():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        subject = data.get('subject', '').strip()
        message = data.get('message', '').strip()

        # Validate all fields
        if not all([name, email, subject, message]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        # Use imported validator from utils
        if not is_valid_email(email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400

        # Admin Email (use env variable)
        admin_msg = (
            f"New message from {name} <{email}>\n\n"
            f"Subject: {subject}\n\n"
            f"Message:\n{message}"
        )

        # User Confirmation
        user_msg = (
            f"Hi {name},\n\n"
            f"Thanks for contacting QuantEdge! We've received your message:\n\n"
            f"\"{message}\"\n\n"
            f"We'll get back to you shortly.\n\n"
            f"– QuantEdge Team"
        )

        # Try sending admin email
        admin_sent = send_email(ADMIN_EMAIL, f"QuantEdge Contact: {subject}", admin_msg)

        if not admin_sent:
            app.logger.error(f"Failed to send admin email for contact: {name}<{email}>")
            return jsonify({
                'success': False,
                'message': 'Failed to send message. Please try later.'
            }), 500

        # Try sending user confirmation
        user_sent = send_email(email, "Thanks for contacting QuantEdge", user_msg)

        if not user_sent:
            app.logger.warning(f"Confirmation failed for: {email}")

        return jsonify({'success': True, 'message': 'Message sent'}), 200

    except Exception:
        app.logger.exception("Contact form error")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500


@app.route('/signup', methods=['POST', 'OPTIONS'])
def signup():
    if request.method == 'OPTIONS':
        return '', 200  # Handle CORS preflight

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'message': 'Invalid JSON'}), 400

    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '').strip()

    if not all([name, email, password]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400

    success, message = create_user(name, email, password)
    return jsonify({'success': success, 'message': message}), (200 if success else 409)


@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    email = data.get('email', '').strip().lower()  # Sanitize

    success, message = logout_user(email)
    return jsonify({'success': success, 'message': message}), (200 if success else 400)


# --- Strategy Code Editor ---
# Ensure CSV file exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'name', 'code', 'timestamp'])
        writer.writeheader()


@app.route('/strategies', methods=['POST', 'GET'])
def strategies():
    if request.method == 'POST':
        data = request.json
        with open(CSV_FILE, 'r') as file:
            reader = list(csv.DictReader(file))
            new_id = int(reader[-1]['id']) + 1 if reader else 1

        data['id'] = new_id
        data['timestamp'] = data.get('timestamp', datetime.utcnow().isoformat())

        with open(CSV_FILE, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['id', 'name', 'code', 'timestamp'])
            writer.writerow(data)

        return jsonify({'message': 'Strategy saved'}), 201

    # GET: Return all strategies
    with open(CSV_FILE, 'r') as file:
        reader = csv.DictReader(file)
        strategies = list(reader)
    return jsonify(strategies)


@app.route('/strategies/<int:id>', methods=['DELETE'])
def delete_strategy(id):
    with open(CSV_FILE, 'r') as file:
        rows = list(csv.DictReader(file))

    updated_rows = [row for row in rows if int(row['id']) != id]

    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'name', 'code', 'timestamp'])
        writer.writeheader()
        writer.writerows(updated_rows)

    return jsonify({'message': f'Strategy {id} deleted'}), 200


# --- Preloaded Data Loader ---
def load_precomputed_data():
    """Load precomputed data using app.logger"""
    global PRELOADED_DATA
    try:
        # Portfolio summary
        portfolio_path = os.path.join(FRONTEND_PATH, "portfolio_summary.json")
        if os.path.exists(portfolio_path):
            with open(portfolio_path) as f:
                PRELOADED_DATA['portfolio_summary'] = json.load(f)
        else:
            app.logger.warning(f"Missing portfolio: {portfolio_path}")

        # Charts directory
        charts_dir = os.path.join(FRONTEND_PATH, "charts")
        if os.path.exists(charts_dir):
            PRELOADED_DATA['tickers'] = [
                f.replace(".json", "")
                for f in os.listdir(charts_dir)
                if f.endswith(".json")
            ]
        else:
            app.logger.warning(f"Missing charts: {charts_dir}")

        # Histogram data
        histogram_path = os.path.join(FRONTEND_PATH, "returns_histogram.json")
        if os.path.exists(histogram_path):
            with open(histogram_path) as f:
                PRELOADED_DATA['returns_histogram'] = json.load(f)
        else:
            app.logger.warning(f"Missing histogram: {histogram_path}")

        # Performance metrics
        metrics_path = os.path.join(FRONTEND_PATH, "performance_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                PRELOADED_DATA['performance_metrics'] = json.load(f)
        else:
            app.logger.warning(f"Missing metrics: {metrics_path}")

    except Exception as e:
        app.logger.error(f"Data loading failed: {str(e)}")


if __name__ == '__main__':
    app.logger.info("Loading precomputed data...")
    load_precomputed_data()

    if PRELOADED_DATA:
        app.logger.info(f"Loaded {len(PRELOADED_DATA.get('tickers', []))} tickers")
    else:
        app.logger.warning("No precomputed data loaded")

    app.logger.info("Starting server on port 5001...")
    app.run(debug=True, port=5001)
