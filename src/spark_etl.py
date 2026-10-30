import os
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize Spark Engine with Kafka Connector
spark = SparkSession.builder \
    .appName("PhishingThreatStreamETL") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Suppress excessive Spark INFO logging in the terminal
spark.sparkContext.setLogLevel("WARN")

# 2. Define the exact JSON schema coming from our producer
schema = StructType([
    StructField("event_id", StringType(), True),
    StructField("raw_text", StringType(), True),
    StructField("true_label", StringType(), True),
    StructField("source", StringType(), True),
    StructField("timestamp", StringType(), True)
])

# 3. Pandas UDF for highly efficient vectorized AI inference
@pandas_udf(ArrayType(FloatType()))
def embed_text(text_series: pd.Series) -> pd.Series:
    # Load the model once per micro-batch execution
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings for the entire batch at once using PyTorch
    embeddings = model.encode(text_series.tolist(), normalize_embeddings=True)
    return pd.Series(embeddings.tolist())

# 4. Define the ChromaDB Loading Logic
def write_to_chroma(df, epoch_id):
    # Convert Spark micro-batch to Pandas
    pdf = df.toPandas()
    if pdf.empty:
        return

    print(f"\n⚡ Processing Micro-Batch {epoch_id} ({len(pdf)} records)...")
    
    # Connect to the ChromaDB Docker container
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection = chroma_client.get_or_create_collection(name="phishing_signatures")

    # Extract columns for vector storage
    ids = pdf["event_id"].astype(str).tolist()
    embeddings = pdf["embedding"].tolist()
    documents = pdf["raw_text"].tolist()
    
    # Pack contextual data (metadata)
    metadatas = pdf.apply(lambda row: {
        "label": row["true_label"], 
        "source": row["source"],
        "timestamp": row["timestamp"]
    }, axis=1).tolist()

    # Upsert into Vector Database
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"✅ Upserted {len(pdf)} vectors to ChromaDB.")

# 5. Build and Start the Streaming Pipeline
def run_pipeline():
    print("🚀 Starting PySpark Vector ETL Stream Engine...")
    
    # Read raw bytes from Redpanda
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "phishing-threat-feed") \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON, extract text, and append the AI vectors
    parsed_stream = raw_stream.selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("embedding", embed_text(col("raw_text")))

    # Output the stream to our custom ChromaDB writer
    query = parsed_stream.writeStream \
        .foreachBatch(write_to_chroma) \
        .outputMode("update") \
        .trigger(processingTime="2 seconds") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_pipeline()
