print('Importing packages')

import pandas as pd
from kafka import KafkaConsumer
import json
from utils.preprocess_util import preprocess  # Import the function directly
from utils.save_to_postgres_util import save_to_postgres

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'fraudulent_transactions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda message: json.loads(message.decode('utf-8'))
)

print('Intialised consumer')


# Process Kafka messages
for message in consumer:
    print(type(message.value))
    # chunk_data = message.value['data']  # Extract chunk data from message
    chunk_data = message.value  # No ['data'] key if it's the raw row dictionary
    chunk_df = pd.DataFrame([chunk_data])  # Wrap in a list to create a DataFrame

    # Preprocess the chunk
    preprocessed_chunk = preprocess(chunk_df)
    
    # Save results to PostgreSQL or further process
    save_to_postgres(preprocessed_chunk)

    print(f"Processed chunk with {len(preprocessed_chunk)} rows")

