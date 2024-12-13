from kafka import KafkaProducer
from faker import Faker
import json
import time

producer = KafkaProducer(
    bootstrap_servers ='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

fake = Faker()

def generate_fake_transaction():
    """Generate fake transaction data."""
    transaction = {
        'transaction_id': fake.uuid4(),
        'amount': fake.random_number(digits=5),
        'card_number': fake.credit_card_number(card_type="mastercard"),
        'is_fraud': fake.boolean(),  # Randomly mark some transactions as fraud
        'timestamp': fake.iso8601()
    }
    return transaction

# Send fake transactions to Kafka every 5 seconds
while True:
    fake_transaction = generate_fake_transaction()
    producer.send('fraudulent_transactions', fake_transaction)  # Send to 'fraudulent_transactions' topic
    print(f"Sent: {fake_transaction}")
    time.sleep(5)  # Wait for 5 seconds before sending the next message
