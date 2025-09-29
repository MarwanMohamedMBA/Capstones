💉 Python Patient Intelligence App

A terminal-based Python application that analyzes immunization data and identifies high-risk patients for vaccine necessity. Designed for healthcare data analysis, reporting, and visualization.

🛠️ Features

Interactive Menu: Choose filters and analysis options for patient datasets.

Data Processing: Reads CSV/JSON files and applies intelligent logic to flag patients needing vaccines.

Reporting: Export results to CSV, JSON, or PDF for actionable insights.

Visualization: Optional charts to track vaccination trends by age group or risk category.

CDC Integration: Supports fetching real-world immunization trends for enhanced accuracy.

💻 Technologies Used

Programming Language: Python 3.x

Libraries: pandas, matplotlib, numpy, fpdf (for PDF export)

File Formats: CSV, JSON, PDF

Tools: Jupyter Notebooks, VS Code, Git/GitHub

📈 Purpose & Impact

This project demonstrates:

End-to-end Python development workflow

Data cleaning, filtering, and conditional logic

File I/O and report generation

Basic data visualization for decision support

Application of domain knowledge in healthcare analytics
=======
# 🩺 Patient Intelligence App

> A complete Python + Azure-powered command-line tool to manage patient vaccine data, identify high-risk patients, and generate smart outreach and reports.

## 🧠 Features

- Import patient data from CSV
- CLI filtering by vaccine type, due date, risk level, age group
- Risk tagging (High/Medium/Low) based on overdue status
- Export filtered lists to CSV, JSON 
- Generate outreach-ready summaries
- Basic charts (matplotlib/seaborn)
- Azure Function endpoint for auto-summary
- Modular Python structure for easy scaling

## 📁 Project Structure

Patient_Intelligence_App/
├── data/ # Input files (patients.csv)
├── exports/ # Output files (csv, json, pdf)
├── patient_app.py # Main CLI script
├── utils.py # Helper functions 
├── README.md # This file
└── requirements.txt # Dependencies


## 🚀 How to Run

```bash
python patient_app.py
