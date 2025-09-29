import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_fake_patients(n=100):
    """
    Generate a DataFrame of fake patient records with random vaccine data.

    Args:
        n (int): Number of fake patient records to generate.

    Returns:
        DataFrame: A pandas DataFrame containing fake patient data with:
            - Patient Name
            - DOB
            - Vaccine Type
            - Last Vaccine
            - Next Due
    """
    vaccines = ["Flu", "COVID", "Hepatitis", "MMR", "Tetanus", "HPV"]
    patients = []

    for _ in range(n):
        dob = fake.date_of_birth(minimum_age=5, maximum_age=90)
        last_vaccine = fake.date_between(start_date='-2y', end_date='today')
        next_due = last_vaccine + timedelta(days=random.randint(180, 730))

        patients.append({
            "Patient Name": fake.name(),
            "DOB": dob.strftime("%Y-%m-%d"),
            "Vaccine Type": random.choice(vaccines),
            "Last Vaccine": last_vaccine.strftime("%Y-%m-%d"),
            "Next Due": next_due.strftime("%Y-%m-%d")
        })

    return pd.DataFrame(patients)

def save_fake_data_csv(filename="mock_patients.csv", count=250):
    """
    Save fake patient data to a CSV file in the `data/` folder.

    Args:
        filename (str): Name of the CSV file to save.
        count (int): Number of fake patient records to generate.
    """
    df = generate_fake_patients(count)
    df.to_csv(f"data/{filename}", index=False)
    print(f"✅ Generated {count} fake patients to data/{filename}")

if __name__ == "__main__":
    save_fake_data_csv()
