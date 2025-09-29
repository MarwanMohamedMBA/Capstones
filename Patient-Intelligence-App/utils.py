"""
🔧 Utility Functions for Patient Intelligence App

Contains:
- Menu display functions (main menu, filter menu)
- Export utilities for filtered data and summary reports
- Banner/header printing
"""

from datetime import datetime
import os
import json
from colorama import Fore


def print_banner():
    """Prints the ASCII app banner to the console."""
    print(Fore.CYAN + """
===================================
  🩺 Patient Intelligence App 🧠
===================================
""")


def main_menu():
    """Displays the main menu options."""
    print(Fore.MAGENTA + "\nMain Menu:")
    print(Fore.CYAN + "1. View all patients")
    print(Fore.CYAN + "2. Filter patients")
    print(Fore.CYAN + "3. Export last filtered results")
    print(Fore.CYAN + "4. Show high-risk patients")
    print(Fore.CYAN + "5. Generate fake patient data")
    print(Fore.CYAN + "6. Generate Summary Report")
    print(Fore.CYAN + "7. Show Charts")
    print(Fore.CYAN + "8. Exit")


def filter_menu():
    """Displays filter options submenu."""
    print(Fore.MAGENTA + "\nFilter Options:")
    print(Fore.CYAN + "a. Filter by vaccine type")
    print(Fore.CYAN + "b. Filter by overdue status")
    print(Fore.CYAN + "c. Filter by age group")
    print(Fore.CYAN + "d. Filter by last vaccine date")
    print(Fore.CYAN + "x. Return to main menu")


def export_to_csv(df):
    """
    Exports the given DataFrame to a timestamped CSV in /exports.

    Args:
        df (pd.DataFrame): The DataFrame to export.
    """
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"exports/export_{timestamp}.csv"
    df.to_csv(file_name, index=False)
    print(Fore.GREEN + f"✅ Exported to {file_name}")


def export_to_json(df):
    """
    Exports the given DataFrame to a timestamped JSON in /exports.

    Args:
        df (pd.DataFrame): The DataFrame to export.
    """
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"exports/export_{timestamp}.json"
    df.to_json(file_name, orient="records", indent=2)
    print(Fore.GREEN + f"✅ Exported to {file_name}")


def export_report_to_txt(report):
    """
    Exports the summary report dictionary to a plain text file.

    Args:
        report (dict): Summary report with counts and breakdowns.
    """
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exports/summary_{timestamp}.txt"
    with open(filename, "w") as f:
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
    print(Fore.GREEN + f"✅ Summary exported to {filename}")


def export_report_to_json(report):
    """
    Exports the summary report dictionary to a JSON file.

    Args:
        report (dict): Summary report with counts and breakdowns.
    """
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exports/summary_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print(Fore.GREEN + f"✅ Summary exported to {filename}")
