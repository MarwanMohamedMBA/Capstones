import pandas as pd
from colorama import Fore, Style, init
import os
from datetime import datetime

init(autoreset=True)
os.makedirs("exports", exist_ok=True)

def print_banner():
    print(Fore.CYAN + """
===================================
  🩺 Patient Intelligence App 🧠
===================================
""")

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

def export_to_csv(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"exports/export_{timestamp}.csv"
    df.to_csv(file_name, index=False)
    print(Fore.GREEN + f"\n✅ Exported to {file_name}")

def export_to_json(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"exports/export_{timestamp}.json"
    df.to_json(file_name, orient="records", indent=2)
    print(Fore.GREEN + f"\n✅ Exported to {file_name}")

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

def main_menu():
    print(Fore.MAGENTA + "\nMain Menu:")
    print(Fore.CYAN + "1. View all patients")
    print(Fore.CYAN + "2. Filter patients")
    print(Fore.CYAN + "3. Export last filtered results")
    print(Fore.CYAN + "4. Show high-risk patients")
    print(Fore.CYAN + "5. Exit")

def filter_menu():
    print(Fore.MAGENTA + "\nFilter Options:")
    print(Fore.CYAN + "a. Filter by vaccine type")
    print(Fore.CYAN + "b. Filter by overdue status")
    print(Fore.CYAN + "c. Filter by age group")
    print(Fore.CYAN + "d. Filter by last vaccine date")
    print(Fore.CYAN + "x. Return to main menu")

def filter_by_vaccine(df):
    vaccine_types = df["Vaccine Type"].dropna().unique()
    dataset_name = getattr(df, 'name', 'dataset')
    print(Fore.MAGENTA + f"\n🧪 Available Vaccines in imported file ({dataset_name}):")
    print(Fore.YELLOW + ", ".join(sorted(vaccine_types)))

    vaccine = input("\nEnter vaccine type to filter by: ").strip()
    filtered = df[df["Vaccine Type"].str.lower() == vaccine.lower()]
    if filtered.empty:
        print(Fore.RED + f"\n❌ No patients found for '{vaccine}'. Please choose from the list above.")
    else:
        print(Fore.YELLOW + f"\n🔎 {len(filtered)} patient(s) found for vaccine '{vaccine}':\n")
        print(filtered.head(10))
    return filtered

def filter_overdue_patients(df):
    today = pd.to_datetime("today").normalize()
    df["Next Due"] = pd.to_datetime(df["Next Due"], errors='coerce')
    filtered = df[df["Next Due"] < today]
    print(Fore.YELLOW + f"\n⚠️ {len(filtered)} patient(s) are overdue for vaccines:\n")
    print(filtered[["Patient Name", "Vaccine Type", "Next Due"]].head(10))
    return filtered

def filter_by_age_group(df):
    df["DOB"] = pd.to_datetime(df["DOB"], errors='coerce')
    today = pd.to_datetime("today").normalize()
    df["Age"] = (today - df["DOB"]).dt.days // 365

    print("\nAge groups:")
    print("1. Under 30")
    print("2. 30–60")
    print("3. Over 60")
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
    
    print(Fore.YELLOW + f"\n👥 {len(filtered)} patient(s) in selected age group:\n")
    print(filtered[["Patient Name", "Age", "Vaccine Type"]].head(10))
    return filtered

def filter_by_last_vaccine_date(df):
    while True:
        date_str = input("Enter a date (YYYY-MM-DD): ").strip()
        try:
            cutoff = pd.to_datetime(date_str)
            filtered = df[pd.to_datetime(df["Last Vaccine"], errors='coerce') >= cutoff]
            print(Fore.YELLOW + f"\n🕒 {len(filtered)} patient(s) had vaccines after {date_str}:\n")
            print(filtered[["Patient Name", "Last Vaccine", "Vaccine Type"]].head(10))
            return filtered
        except Exception as e:
            print(Fore.RED + "❌ Invalid date format. Try again.")
            continue

# ---------- MAIN APP ----------

def main():
    print_banner()
    last_filtered = None

    file_name = input("Enter CSV file name (default: mock_patients.csv): ").strip()
    if not file_name:
        file_name = "mock_patients.csv"

    df = load_data(file_name)
    if df is None:
        return

    df.name = file_name
    df = tag_risk_level(df)

    while True:
        main_menu()
        choice = input("Choose an option: ").strip()

        if choice == '1':
            print(Fore.YELLOW + "\n📋 First 10 patients:\n")
            print(df.head(10))

        elif choice == '2':
            while True:
                filter_menu()
                sub = input("Choose a filter option: ").strip().lower()
                if sub == 'a':
                    last_filtered = filter_by_vaccine(df)
                elif sub == 'b':
                    last_filtered = filter_overdue_patients(df)
                elif sub == 'c':
                    last_filtered = filter_by_age_group(df)
                elif sub == 'd':
                    last_filtered = filter_by_last_vaccine_date(df)
                elif sub == 'x':
                    break
                else:
                    print(Fore.RED + "❌ Invalid selection.")

        elif choice == '3':
            if last_filtered is None or last_filtered.empty:
                print(Fore.RED + "❌ No filtered results to export.")
            else:
                export_type = input("Export as CSV or JSON? ").strip().lower()
                if export_type == "csv":
                    export_to_csv(last_filtered)
                elif export_type == "json":
                    export_to_json(last_filtered)
                else:
                    print(Fore.RED + "❌ Invalid format. Type 'csv' or 'json'.")

        elif choice == '4':
            high_risk = df[df["Risk Level"] == "High"]
            print(Fore.RED + f"\n⚠️ {len(high_risk)} High-Risk Patients:\n")
            print(high_risk[["Patient Name", "Age", "Next Due", "Risk Level"]].head(10))

        elif choice == '5':
            print(Fore.GREEN + "👋 Goodbye.")
            break

        else:
            print(Fore.RED + "❌ Invalid option. Try again.")

if __name__ == "__main__":
    main()
