# Privacy Notice

## What this tool is
This is a research prototype developed as a final year university project. It is **not approved for clinical use** and must not be used to inform clinical decisions.

## What data is collected
The following information may be entered and stored locally on the device running this application:

- A **case reference identifier** — this must be an anonymised code only. Do not enter patient names, NHS numbers, dates of birth or any other personally identifiable information.
- A visit date
- Which eye is being assessed
- Computed measurements derived from the uploaded image (area, diameter, zone, opacity score)
- An auto-generated documentation note

## What data is NOT collected
- Patient names or any directly identifiable information
- The uploaded image itself (only a short hash fingerprint is stored to prevent duplicate entries)
- Any data is sent to external servers beyond what is described below

## Third party services
This tool uses the **Groq API** to generate clinical documentation notes. A summary of computed measurements (no images, no patient identifiers) is sent to Groq's servers for this purpose. Groq's privacy policy is available at https://groq.com/privacy-policy.

## Where data is stored
All case and visit data is stored in a local SQLite database file at `data_store/app.db` on the device running the application. This file is not transmitted anywhere. It is the responsibility of the person operating this tool to ensure the device and database file are appropriately secured.

## Data deletion
To delete all stored data, delete the file `data_store/app.db`. Individual visits cannot currently be deleted through the UI.

## Data retention
There is no automatic data retention or deletion policy in this prototype. Operators should delete the database file when it is no longer needed.

## GDPR
This tool is a research prototype used for academic purposes. If it were to be deployed in a real clinical setting, a full Data Protection Impact Assessment (DPIA) would be required, along with a lawful basis for processing, appropriate consent mechanisms and compliance with UK GDPR and the Data Protection Act 2018.

## Contact
This tool was developed by Douaa Ghezali as part of a final year project at the University of Birmingham.