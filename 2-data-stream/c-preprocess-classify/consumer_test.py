from kafka import KafkaConsumer
import json

# Kafka consumer configuration
consumer = KafkaConsumer(
    'fraudulent_transactions',           # The topic to subscribe to
    bootstrap_servers='localhost:9092',  # Kafka server address
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),  # Deserialize the messages from JSON
    # group_id='fraud_consumer_group'      # Consumer group (optional)
)

print("Consuming messages from 'fraudulent_transactions' topic...")

# Consume messages (runs indefinitely)
for message in consumer:
    print(f"Received message: {message.value}")
