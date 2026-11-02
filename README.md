# Automated Data Reconciliation & Analytics Framework

A practical reconciliation and analytics prototype that I developed as part of my M.Tech Software Engineering dissertation at BITS Pilani WILP.

## Overview

I developed this application to simplify the review of exported booking reports by automating key data quality checks and presenting the results through an interactive Streamlit dashboard.

## Features

- Upload Master Report CSV files
- Identify missing or null values
- Detect duplicate TRAVEL ticket numbers
- Identify ticket series breaks or missing tickets
- Check Net Sale discrepancies
- Flag online transactions with missing Order IDs
- Visualize sales analytics and discrepancies through charts
- Download the validated reconciliation report

## Technology Stack

- Python
- Streamlit
- Pandas
- Plotly
- OpenPyXL

## Run Locally

1. Install Python.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL displayed in the terminal.

## Scope and Limitations

This is a report based prototype that I developed to process uploaded CSV reports. It does not connect directly to the live booking system or its database.

For demonstrations, I used anonymized or sample data. (Test Data Only for Demo Purpose)

## Project Context

**Dissertation:** Automated Data Reconciliation & Analytics Framework  
**Programme:** M.Tech Software Engineering, BITS Pilani WILP

