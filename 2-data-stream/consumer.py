from kafka import KafkaConsumer
import json


consumer = KafkaConsumer(
    'fraudulent_transactions',
    value_serializer=lambda x: json.loads(x.decode('utf-8'))
)

# list for messages on the topic

for message in consumer:
    transaction = message.value
    print(f"Received: {transaction}")
    ## process here 