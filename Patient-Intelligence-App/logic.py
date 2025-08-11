import pandas as pd
from colorama import Fore
from datetime import datetime
import os

def load_data(file_name):
    full_path = f"data/{file_name}"
    if not os.path.exists(full_path):
        print(Fore.RED + f"❌ File not found: {full_path}")
        return None
    try:
        df = pd.read_csv(full_path)
        required_cols = {"Patient Name", "DOB", "Next Due", "Vaccine Type"}
        if not required_cols.issubset(df.columns):
            print(Fore.RED + "❌ Missing required columns in file.")
            return None
        print(Fore.GREEN + f"✅ Loaded {len(df)} records from {full_path}")
        print(Fore.YELLOW + f"🧾 Columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        print(Fore.RED + f"❌ Failed to load data: {e}")
        return None

def tag_risk_level(df):
    today = pd.to_datetime("today").normalize()
    df["Next Due"] = pd.to_datetime(df["Next Due"], errors="coerce")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df["Age"] = (today - df["DOB"]).dt.days // 365

    def get_risk(row):
        overdue_days = (today - row["Next Due"]).days
        age = row["Age"]
        if overdue_days > 365 or age >= 70:
            return "High"
        elif overdue_days > 0 or 40 <= age < 70:
            return "Medium"
        else:
            return "Low"

    df["Risk Level"] = df.apply(get_risk, axis=1)
    print(Fore.GREEN + "✅ Risk levels tagged.")
    return df

def filter_by_vaccine(df):
    vaccine_types = df["Vaccine Type"].dropna().unique()
    dataset_name = getattr(df, 'name', 'dataset')
    print(Fore.MAGENTA + f"\n🧪 Available Vaccines in imported file ({dataset_name}):")
    print(Fore.YELLOW + ", ".join(sorted(vaccine_types)))

    vaccine = input("\nEnter vaccine type to filter by: ").strip()
    filtered = df[df["Vaccine Type"].str.lower() == vaccine.lower()]
    if filtered.empty:
        print(Fore.RED + f"\n❌ No patients found for '{vaccine}'.")
    else:
        print(Fore.YELLOW + f"\n🔎 {len(filtered)} patient(s) found for vaccine '{vaccine}':\n")
        print(filtered.head(10))
    return filtered

def filter_overdue_patients(df):
    today = pd.to_datetime("today").normalize()
    df["Next Due"] = pd.to_datetime(df["Next Due"], errors='coerce')
    filtered = df[df["Next Due"] < today]
    print(Fore.YELLOW + f"\n⚠️ {len(filtered)} patient(s) are overdue:\n")
    print(filtered[["Patient Name", "Vaccine Type", "Next Due"]].head(10))
    return filtered

def filter_by_age_group(df):
    df["DOB"] = pd.to_datetime(df["DOB"], errors='coerce')
    today = pd.to_datetime("today").normalize()
    df["Age"] = (today - df["DOB"]).dt.days // 365

    print("\nAge groups:\n1. Under 30\n2. 30–60\n3. Over 60")
    group = input("Choose an age group (1–3): ").strip()

    if group == '1':
        filtered = df[df["Age"] < 30]
    elif group == '2':
        filtered = df[(df["Age"] >= 30) & (df["Age"] <= 60)]
    elif group == '3':
        filtered = df[df["Age"] > 60]
    else:
        print(Fore.RED + "❌ Invalid selection.")
        return pd.DataFrame()
    
    print(Fore.YELLOW + f"\n👥 {len(filtered)} patient(s) in selected group:\n")
    print(filtered[["Patient Name", "Age", "Vaccine Type"]].head(10))
    return filtered

def filter_by_last_vaccine_date(df):
    try:
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
        cutoff = pd.to_datetime(date_str)
        filtered = df[pd.to_datetime(df["Last Vaccine"]) >= cutoff]
        print(Fore.YELLOW + f"\n🕒 {len(filtered)} patients since {date_str}:\n")
        print(filtered[["Patient Name", "Last Vaccine", "Vaccine Type"]].head(10))
        return filtered
    except:
        print(Fore.RED + "❌ Invalid date format.")
        return pd.DataFrame()

def generate_summary_report(df):
    today = pd.to_datetime("today").normalize()
    df["Next Due"] = pd.to_datetime(df["Next Due"], errors='coerce')

    total = len(df)
    overdue = len(df[df["Next Due"] < today])
    vaccine_counts = df["Vaccine Type"].value_counts().to_dict()
    risk_counts = df["Risk Level"].value_counts().to_dict()

    report = {
        "Total Patients": total,
        "Overdue Patients": overdue,
        "Vaccine Breakdown": vaccine_counts,
        "Risk Levels": risk_counts
    }

    print(Fore.YELLOW + "\n📊 Summary Report:")
    for key, value in report.items():
        print(f"{key}: {value}")

    return report
