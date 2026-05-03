"""
Sandbox Data Generator for GridPulse AI
Pre-generates the 'Normal' vs 'Theft' power profiles for the live demo.
"""

import os
import sys
from datetime import datetime, timedelta

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from simulation.data_generator import SyntheticDatasetGenerator

def main():
    print("Initializing GridPulse Sandbox...")
    generator = SyntheticDatasetGenerator()
    
    # Generate 1 week of data ending today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    dataset = generator.generate_dataset(
        start_date=start_date,
        end_date=end_date,
        theft_percentage=0.2,  # 20% meters have theft
        include_data_gaps=True,
        gap_percentage=0.02
    )
    
    # Save dataset to standard location used by dashboard
    output_dir = os.path.join(os.getcwd(), "data", "synthetic", "sandbox")
    os.makedirs(output_dir, exist_ok=True)
    generator.save_dataset(output_dir)
    print(f"Sandbox data generation complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()
