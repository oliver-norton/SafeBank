### DATA CONNECTOR 

# load test_transaction
# load test_identity 

# pull data from google drive API 
# select just column by column 

# Merge the dataframes
TstIdTr = pd.merge(trainTransaction, trainIdentity, on='TransactionID', how='left')

# verify columns match training set:

# Load the training column names from the saved file
with open('trIdTr_columns.txt', 'r') as f:
    loaded_columns = f.read().splitlines()

# Convert both lists to sets
set_TstIdTr = set(tsIdTr.columns)
set_loaded_columns = set(loaded_columns)

# Find columns in trIdTr but not in loaded_columns
missing_in_loaded = TstIdTr - set_loaded_columns

# Find columns in loaded_columns but not in TstIdTr
missing_in_TstIdTr = set_loaded_columns - set_TstIdTr

# assert if either of above sets are both null, if not, print them and say missing columns

# results in TstTrId.csv 
# stored in GoogleDrive with Google drive api 