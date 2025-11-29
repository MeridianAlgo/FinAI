from googleapiclient.discovery import build
from google.oauth2 import service_account
import os
from datetime import datetime

SERVICE_ACCOUNT_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Use your spreadsheet ID
SPREADSHEET_ID = '1TMiX9YDSH7ifm5MizBEYRcTaojBCrnWF1roqI8qulDI'

# Replace 'Sheet1' with the actual tab name in your sheet
RANGE_NAME = 'Sheet1!A1'

def update_google_sheets(dataset_name, status="completed"):
"""Update Google Sheets when a dataset status changes"""
try:
# Check if credentials file exists
if not os.path.exists(SERVICE_ACCOUNT_FILE):
print(f"Warning: {SERVICE_ACCOUNT_FILE} not found. Skipping Google Sheets update.")
return False

# Authenticate and build service
creds = service_account.Credentials.from_service_account_file(
SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('sheets', 'v4', credentials=creds)

# Get current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Prepare the row data
values = [
[dataset_name, status, timestamp]
]

body = {'values': values}

# Append to the sheet (adds new row)
result = service.spreadsheets().values().append(
spreadsheetId=SPREADSHEET_ID,
range=RANGE_NAME,
valueInputOption="RAW",
body=body
).execute()

updated_cells = result.get('updates', {}).get('updatedCells', 0)
print(f"Google Sheets updated: {updated_cells} cells added for dataset '{dataset_name}'")
return True

except Exception as e:
print(f"Error updating Google Sheets: {str(e)}")
return False

def initialize_sheet():
"""Initialize the Google Sheet with headers if it's empty"""
try:
if not os.path.exists(SERVICE_ACCOUNT_FILE):
print(f"Warning: {SERVICE_ACCOUNT_FILE} not found. Cannot initialize sheet.")
return False

# Authenticate and build service
creds = service_account.Credentials.from_service_account_file(
SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('sheets', 'v4', credentials=creds)

# Check if sheet has data
result = service.spreadsheets().values().get(
spreadsheetId=SPREADSHEET_ID,
range=RANGE_NAME
).execute()

# If sheet is empty, add headers
if not result.get('values', []):
headers = [["Dataset Name", "Status", "Timestamp"]]
body = {'values': headers}

result = service.spreadsheets().values().update(
spreadsheetId=SPREADSHEET_ID,
range=RANGE_NAME,
valueInputOption="RAW",
body=body
).execute()

print("Google Sheets initialized with headers")
return True
else:
print("Google Sheets already has data")
return True

except Exception as e:
print(f"Error initializing Google Sheets: {str(e)}")
return False

if __name__ == "__main__":
# Test the functions
print("Testing Google Sheets integration...")
initialize_sheet()
update_google_sheets("test_dataset", "testing")
