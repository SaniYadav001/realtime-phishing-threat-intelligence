import csv
import json
import time
from datetime import datetime
from kafka import KafkaProducer

# Initialize Kafka Producer pointing to our local Redpanda container
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC_NAME = 'phishing-threat-feed'

def stream_data():
    print(f"🚀 Starting live threat stream to topic: {TOPIC_NAME}...")
    
    with open('data/baseline_threats.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Create a JSON payload mimicking a real-time event
            payload = {
                "event_id": row["id"],
                "raw_text": row["text"],
                "true_label": row["label"],
                "source": row["source"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to Redpanda
            producer.send(TOPIC_NAME, payload)
            print(f"[{payload['timestamp']}] Sent -> {payload['raw_text'][:50]}...")
            
            # Pause for 2 seconds to simulate real-time traffic
            time.sleep(2)
            
    producer.flush()
    print("✅ Stream complete.")

if __name__ == "__main__":
    stream_data()
