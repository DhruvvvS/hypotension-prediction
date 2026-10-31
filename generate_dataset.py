import numpy as np
import pandas as pd
import os

# Generates a medically realistic intraoperative MAP dataset.
# Mimics real physiological responses: baseline drift, induction drops, and recovery.
def generate_realistic_iOH_dataset(n_patients=600, seed=42):
        
    np.random.seed(seed)
    records = []
    
    print(f"Generating realistic data for {n_patients} patients...")
    
    for case_id in range(n_patients):
        # Realistic surgery duration: 120 to 240 minutes
        duration = np.random.randint(120, 240)
        minutes = np.arange(duration)
        
        # Realistic patient age: 25 to 85
        age = int(np.random.normal(60, 15))
        age = np.clip(age, 25, 85)
        
        # Baseline MAP: normally distributed around 85-95 mmHg
        baseline_map = np.random.normal(90, 8)
        
        # Natural physiological drift (random walk) + slight downward trend over long surgeries
        drift = np.cumsum(np.random.normal(0, 0.8, duration)) - 0.05 * minutes
        
        # Base signal with noise
        map_values = baseline_map + drift + np.random.normal(0, 1.5, duration)
        
        # Inject realistic hypotensive events (e.g., anesthesia induction, blood loss)
        # 40% of patients will have at least one significant hypotensive event
        if np.random.rand() < 0.40:
            num_events = np.random.randint(1, 3)
            for _ in range(num_events):
                # Event starts between minute 10 and (duration - 20)
                start_min = np.random.randint(10, max(11, duration - 20))
                # Event lasts between 2 and 8 minutes (clinically significant)
                event_duration = np.random.randint(2, 9)
                end_min = min(start_min + event_duration, duration)
                
                # Drop MAP to between 55 and 64 mmHg
                drop_amount = np.random.uniform(25, 40)
                map_values[start_min:end_min] -= drop_amount
                
                # Add a gradual recovery slope after the event
                recovery_len = min(5, duration - end_min)
                if recovery_len > 0:
                    map_values[end_min:end_min+recovery_len] += np.linspace(drop_amount*0.5, 0, recovery_len)

        # Ensure MAP doesn't go below physiologically possible limits (e.g., 40) or above 150
        map_values = np.clip(map_values, 40, 150)
        
        # Create records for this patient
        for t in range(duration):
            records.append({
                "case_id": case_id,
                "minute": t,
                "map": round(map_values[t], 1),
                "age": age
            })
            
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    # 1. Generate the data
    df = generate_realistic_iOH_dataset(n_patients=600)
    
    # 2. Ensure the data/raw directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # 3. Save to CSV
    output_path = "data/raw/intraoperative_map_data.csv"
    df.to_csv(output_path, index=False)
    
    # 4. Print summary statistics to prove it's realistic
    total_events = df[df['map'] < 65].groupby('case_id').size().shape[0]
    print(f"Dataset generated successfully!")
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Total patients: {df['case_id'].nunique()}")
    print(f"Patients with hypotensive events (MAP < 65): {total_events} ({total_events/600*100:.1f}%)")
    print(f"Mean MAP: {df['map'].mean():.1f} mmHg | Std: {df['map'].std():.1f} mmHg")