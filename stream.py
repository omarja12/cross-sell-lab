"""Replay the held-out customers through Kafka, score them with Spark Structured Streaming,
and check every streamed score against batch scoring."""
import glob
import time

import joblib
import pandas as pd
import pyarrow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lab import features

KAFKA = "localhost:9092"
topic = f"customers_{int(time.time())}"  # fresh topic per run, so a rerun can't double-count
out, checkpoint = f"data/scores/{topic}", f"data/checkpoints/{topic}"

test = pd.read_csv("data/test.csv")
model = joblib.load("data/model.joblib")
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"env: spark {spark.version}, pyarrow {pyarrow.__version__}")

# Producer: each customer becomes one JSON message.
customers = spark.createDataFrame(test)
messages = customers.select(F.to_json(F.struct("*")).alias("value"))
n, size = messages.select(F.count("*"), F.sum(F.octet_length("value"))).first()
print(f"produced: {n:,} JSON messages, {size:,} bytes ({size / n:.0f} bytes each on average)")
messages.write.format("kafka").option("kafka.bootstrap.servers", KAFKA).option("topic", topic).save()

on_broker = spark.read.format("kafka").option("kafka.bootstrap.servers", KAFKA).option("subscribe", topic).load()
print("kafka: " + ", ".join(f"partition {r['partition']} holds {r['count']:,} messages"
                            for r in on_broker.groupBy("partition").count().orderBy("partition").collect()))


def score(batches):
    for b in batches:
        yield pd.DataFrame({"id": b["id"], "score": model.predict_proba(features(b))[:, 1]})


# Consumer: read the topic in micro-batches of 10,000 messages, score with the same code as lab.py.
query = (spark.readStream.format("kafka")
         .option("kafka.bootstrap.servers", KAFKA).option("subscribe", topic)
         .option("startingOffsets", "earliest").option("maxOffsetsPerTrigger", 10_000).load()
         .select(F.from_json(F.col("value").cast("string"), customers.schema).alias("c")).select("c.*")
         .mapInPandas(score, "id long, score double")
         .writeStream.format("parquet").option("path", out).option("checkpointLocation", checkpoint)
         .trigger(availableNow=True).start())
query.awaitTermination()
batches = [p["numInputRows"] for p in query.recentProgress if p["numInputRows"]]
print(f"consumed: {len(batches)} micro-batches of {batches} rows")
print(f"sink: {len(glob.glob(f'{out}/*.parquet'))} parquet files")

got = spark.read.parquet(out).toPandas().sort_values("id").reset_index(drop=True)
want = pd.DataFrame({"id": test["id"], "score": model.predict_proba(features(test))[:, 1]}).sort_values("id").reset_index(drop=True)

assert got["id"].equals(want["id"]), "every customer scored exactly once"
diff = (got["score"] - want["score"]).abs().max()
assert diff < 1e-12, f"stream and batch disagree by {diff}"
print(f"{len(got):,} customers streamed through Kafka; max |stream - batch| = {diff:.1e}")
