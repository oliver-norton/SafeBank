from kafka import KafkaConsumer
import json

# Initialize Kafka consumer to listen on 'fraudulent_transactions' topic
consumer = KafkaConsumer(
    'fraudulent_transactions',
    bootstrap_servers='localhost:9092',  # Kafka server address
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # Deserialize the message back to JSON
)

# Listen for messages on the Kafka topic
for message in consumer:
    transaction = message.value
    print(f"Received: {transaction}")
    # You can now process the transaction here (e.g., classify fraud)