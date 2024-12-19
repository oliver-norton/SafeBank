import pandas as pd
from kafka import KafkaProducer
import json
import os
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda raw_message: json.dumps(raw_message).encode('utf-8'),
    buffer_memory=33554432  # 32MB buffer (default is 32MB)
)

# Paths to your data files
# base_path = '../../1-model/for_g_drive/tsIdTr_base.pkl's
# synth_path = '../../1-model/for_g_drive/TsIdTr_synth.pkl'

base_path = '/home/watoomi/SafeBank/1-model/for_g_drive/tsIdTr_base.pkl'
synth_path = '/home/watoomi/SafeBank/1-model/for_g_drive/TsIdTr_synth.pkl'


print("Base path: ", os.path.abspath(base_path))
print("Synth path: ", os.path.abspath(synth_path))

print('Allow 15 seconds between messages')


# location of producer_chunks.py

# SafeBank/2-data-stream/b_stream_data/producer_chunks.py

# location of pkl files:

# SafeBank/1-model/for_g_drive/tsIdTr_base.pkl
# SafeBank/1-model/for_g_drive/TsIdTr_synth.pkl


# Read files in chunks of 1000
# takes actual test data and 
base_chunks = pd.read_pickle(base_path)
synth_chunks = pd.read_pickle(synth_path)



rows_per_chunk = 20
combined = pd.concat([base_chunks[:rows_per_chunk], synth_chunks[:rows_per_chunk]])
for idx, row in combined.iterrows():
    producer.send('fraudulent_transactions', row.to_dict())  # Send data to Kafka
    print(f"Sent row {idx}: {row.to_dict()}")
    time.sleep(15)  # 15 seconds delay