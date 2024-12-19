import psycopg2
import pandas as pd
from sqlalchemy import create_engine

def save_to_postgres(df):
    try:
        # Establishing the connection using psycopg2 and sqlalchemy
        # Replace 'your_password' with your actual PostgreSQL password
        engine = create_engine('postgresql://postgres:12345678@localhost:5432/SafeBank_db')
        
        # Storing the DataFrame in the 'fraud_classifications' table in the PostgreSQL database
        # If the table exists, it will be overwritten; set if_exists='replace' to 'append' if you want to add data instead
        df.to_sql('fraud_classifications', engine, index=False, if_exists='append', method='multi')
        
        print("Data saved to PostgreSQL successfully!")
    
    except Exception as e:
        print(f"Error saving data to PostgreSQL: {e}")

# Sample usage:
# Assuming 'preprocessed_data' is your processed DataFrame
# save_to_postgres(preprocessed_data)
