#!/bin/bash
# GridPulse AI - One-Click Setup & Launch Script
# Ensures all dependencies are installed and launches the Mission Control Dashboard

echo "⚡ Initializing GridPulse AI Sovereign Environment..."

# Check if python is available
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python3."
    exit
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Generate synthetic sandbox data
echo "⚙️ Generating Sandbox Data (Normal vs Theft profiles)..."
python3 generate_sandbox.py

# Launch the Streamlit Mission Control Dashboard
echo "🚀 Launching GridPulse AI Mission Control..."
streamlit run src/dashboard/app.py
