import psycopg2
from psycopg2 import sql

def create_database():
    conn = None
    cursor = None
    try:
        # Connect to the default 'postgres' database to create a new one
        conn = psycopg2.connect(
            dbname="postgres",  # Default database
            user="postgres",  # Replace with your PostgreSQL username
            password="12345678",  # Replace with your PostgreSQL password
            host="localhost",
            port="5432"
        )
        conn.autocommit = True  # Allows us to create the DB without needing to commit manually
        cursor = conn.cursor()

        # SQL to create a new database
        create_db_query = sql.SQL("CREATE DATABASE {db_name}").format(
            db_name=sql.Identifier("SafeBank_db")  # Replace with your desired database name
        )
        cursor.execute(create_db_query)
        print("Database 'fraud_detection' created successfully!")

    except Exception as e:
        print(f"Error creating database: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Call the function to create the database
create_database()
