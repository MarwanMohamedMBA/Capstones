"""
📦 Patient Intelligence CLI App

Main entry point for running the Patient Intelligence application.
This app allows users to:
- Load and validate a patient vaccination dataset
- Tag patients by risk level based on age and overdue vaccine status
- Apply filters (age group, vaccine type, overdue, last dose date)
- Export filtered data to CSV or JSON
- Generate and export summary reports
- Plot charts by age group and vaccine gaps
- Generate fake mock data for testing
"""

from logic import (
    load_data,
    tag_risk_level,
    filter_by_vaccine,
    filter_overdue_patients,
    filter_by_age_group,
    filter_by_last_vaccine_date,
    generate_summary_report
)

from utils import (
    print_banner,
    main_menu,
    filter_menu,
    export_to_csv,
    export_to_json,
    export_report_to_txt,
    export_report_to_json
)

from charts import plot_vaccine_gaps, plot_age_groups
from generate_fake_data import save_fake_data_csv
from colorama import Fore
import pandas as pd


def main():
    """
    Main CLI loop for interacting with the patient app.
    Handles loading data, showing menus, applying filters,
    exporting results, generating reports, and visualizing charts.
    """
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
            count_str = input("How many fake patients to generate? (default: 250): ").strip()
            count = int(count_str) if count_str.isdigit() else 250
            save_fake_data_csv(count=count)

        elif choice == '6':
            report = generate_summary_report(df)
            format_choice = input("Export as TXT or JSON? ").strip().lower()
            if format_choice == "txt":
                export_report_to_txt(report)
            elif format_choice == "json":
                export_report_to_json(report)
            else:
                print(Fore.RED + "❌ Invalid format. Type 'txt' or 'json'.")

        elif choice == '7':
            plot_vaccine_gaps(df)
            plot_age_groups(df)

        elif choice == '8':
            print(Fore.GREEN + "👋 Goodbye.")
            break

        else:
            print(Fore.RED + "❌ Invalid option. Try again.")


if __name__ == "__main__":
    main()
