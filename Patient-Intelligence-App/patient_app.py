import pandas as pd
from colorama import Fore, Style, init
import os

# Initialize colorama
init(autoreset=True)

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
        print(Fore.GREEN + f"✅ Loaded {len(df)} records from {full_path}")
        print(Fore.YELLOW + f"🧾 Columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        print(Fore.RED + f"❌ Failed to load data: {e}")
        return None

def main_menu():
    print(Fore.MAGENTA + "\nMain Menu:")
    print(Fore.CYAN + "1. View all patients")
    print(Fore.CYAN + "2. Filter patients (coming soon)")
    print(Fore.CYAN + "3. Export results (coming soon)")
    print(Fore.CYAN + "4. Exit")


def main():
    print_banner()

    file_name = input("Enter CSV file name, must be in data directory (default: mock_patients.csv): ").strip()
    if not file_name:
        file_name = "mock_patients.csv"

    df = load_data(file_name)
    if df is None:
        return

    while True:
        main_menu()
        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            print(Fore.YELLOW + "\n📋 First 10 patients:\n")
            print(df.head(10))
        elif choice == '4':
            print(Fore.GREEN + "👋 Goodbye.")
            break
        else:
            print(Fore.BLUE + "🚧 That feature is under construction.")

if __name__ == "__main__":
    main()
