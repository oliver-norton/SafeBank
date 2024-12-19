import pandas as pd
from kafka import KafkaConsumer
import json

from utils import preprocess_util as preprocess
from utils import save_to_postgres_util as save_to_postgres

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'fraudulent_transactions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)


# Process Kafka messages
for message in consumer:
    chunk_data = message.value['data']  # Extract chunk data from message
    chunk_df = pd.DataFrame(chunk_data)  # Convert chunk to a DataFrame

    # Preprocess the chunk
    preprocessed_chunk = preprocess(chunk_df)
    
    # Save results to PostgreSQL or further process
    save_to_postgres(preprocessed_chunk)

    print(f"Processed chunk with {len(preprocessed_chunk)} rows")

