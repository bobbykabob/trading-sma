#!/bin/bash

echo "Installing required packages..."
pip install -r requirements.txt

echo "Starting the trading dashboard..."
streamlit run app.py
