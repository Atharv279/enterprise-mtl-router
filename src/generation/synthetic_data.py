import random
import numpy as np
from faker import Faker
import pandas as pd
from datetime import datetime, timedelta
from uuid import uuid4
import os

# Initialize multilingual Faker providers
fake = Faker(['en_US', 'es_ES', 'hi_IN'])

# Define correlation mapping
PRIORITY_MAPPING = {
    "High": {
        "keywords": ["hazardous", "immediate", "collapsed", "breach", "injury", "critical"],
        "eta_mean": 1.5,
        "eta_std": 0.5
    },
    "Medium": {
        "keywords": ["delayed", "maintenance", "intermittent", "pending", "repair"],
        "eta_mean": 5.0,
        "eta_std": 2.0
    },
    "Low": {
        "keywords": ["inquiry", "update", "routine", "administrative", "question"],
        "eta_mean": 21.0,
        "eta_std": 5.0
    }
}

def generate_synthetic_complaints(num_records=1000):
    dataset = []
    
    for _ in range(num_records):
        # 1. Determine ground truth priority stochastically
        priority = random.choices(["High", "Medium", "Low"], weights=[0.15, 0.35, 0.50])[0]
        params = PRIORITY_MAPPING[priority]
        
        # 2. Correlate text with priority
        keyword = random.choice(params["keywords"])
        raw_text = f"{fake.sentence()} The situation is {keyword}. {fake.text()}"
        
        # 3. Correlate ETA with priority (Gaussian distribution)
        eta = max(0.5, np.random.normal(params["eta_mean"], params["eta_std"]))
        
        # 4. Map to integer classification for the MTL Network (1=High, 2=Medium, 3=Low)
        priority_int = 1 if priority == "High" else (2 if priority == "Medium" else 3)
        
        record = {
            "complaint_id": str(uuid4()),
            "timestamp": fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
            "raw_text": raw_text,
            "predicted_priority": priority_int,
            "estimated_eta": round(eta, 2),
            "language": fake.current_country_code()
        }
        dataset.append(record)
        
    return pd.DataFrame(dataset)

if __name__ == "__main__":
    df = generate_synthetic_complaints(5000)
    
    # 1. Dynamically find the absolute path to the project root
    # __file__ is src/generation/synthetic_data.py
    # dirname x3 moves up to the complaint_router/ directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 2. Construct the exact output path
    output_dir = os.path.join(project_root, "data", "synthetic")
    output_path = os.path.join(output_dir, "training_data.json")
    
    # 3. Create the directories if they don't exist yet
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Save the file
    df.to_json(output_path, orient="records")
    print(f"Generated {len(df)} synthetic records.")
    print(f"Successfully saved to: {output_path}")