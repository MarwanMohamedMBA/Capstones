import logging
import azure.functions as func
import pandas as pd
import json
from datetime import datetime

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("⚙️ generateSummary function processed a request.")

    try:
        df = pd.read_csv('data/mock_patients.csv')

        # Convert date columns
        df['last_vaccine'] = pd.to_datetime(df['last_vaccine'], errors='coerce')
        df['next_due'] = pd.to_datetime(df['next_due'], errors='coerce')

        # Filter patients with upcoming vaccines within 30 days
        today = pd.Timestamp(datetime.today().date())
        soon_due = df[
            (df['next_due'].notna()) &
            (df['next_due'] >= today) &
            (df['next_due'] <= today + pd.Timedelta(days=30))
        ]

        # Format response
        summary = [
            {
                "Patient Name": row["name"],
                "Vaccine Type": row["vaccine_type"],
                "Next Due": row["next_due"].strftime("%Y-%m-%d")
            }
            for _, row in soon_due.iterrows()
        ]

        return func.HttpResponse(
            json.dumps(summary, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"❌ Failed to generate summary: {e}")
        return func.HttpResponse(
            "Failed to generate summary.",
            status_code=500
        )
