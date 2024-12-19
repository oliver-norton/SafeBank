import psycopg2

def create_table():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            dbname="SafeBank_db",
            user="postgres",
            password="12345678",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        # SQL to create the table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS fraud_classifications (
            id SERIAL PRIMARY KEY,
            TransactionID BIGINT NOT NULL,
            isFraud_Prediction INTEGER NOT NULL,
            Fraud_Probability NUMERIC(5, 3) NOT NULL,
            Classification_Time TIMESTAMP NOT NULL,
            Model_Version TEXT NOT NULL,
            Prediction_Threshold NUMERIC(5, 3) NOT NULL,
            Model_Name TEXT NOT NULL
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        print("Table 'fraud_classifications' created successfully!")

    except Exception as e:
        print(f"Error creating table: {e}")

    finally:
        cursor.close()
        conn.close()

# Call the function to create the table
create_table()
