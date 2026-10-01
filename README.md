# cross-sell-lab

An insurer has 381,109 health-insurance customers. Which of them should it call to sell car insurance?

This repo answers that twice:

- **as a model**, with an honest evaluation next to the two most common ways of fooling yourself;
- **as a stream**, with customers arriving one message at a time through Kafka and Spark scoring them, using the same feature code and the same model as the batch job.

Data: Kaggle's [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) (`train.csv`, 12.3% bought). Not redistributed here.

## 1. The model, and two ways to flatter it

Gradient-boosted trees (`HistGradientBoostingClassifier`). Region and sales channel are treated as categories, not numbers. Test set: 30% of customers, stratified, never touched during training.

| protocol | rows | buy rate | ROC AUC | PR AUC | precision@0.5 | recall@0.5 | precision, top 10% |
|---|---|---|---|---|---|---|---|
| honest | 114,333 | 0.123 | 0.858 | 0.371 | 0.471 | 0.003 | 0.398 |
| A: outliers dropped from test | 63,862 | 0.143 | 0.838 | 0.370 | 0.432 | 0.002 | 0.393 |
| B: oversampled before split | 200,640 | 0.500 | 0.863 | 0.808 | 0.743 | 0.936 | 0.850 |

**How to read it:**

- **Honest.** At the usual 0.5 cut-off the model calls almost no one, because at a 12% buy rate few customers ever score above 0.5. The model is still useful: calling the top 10% of its ranking converts **39.8%** of calls, 3.2× the 12.3% base rate. **The ranking is the product, not the 0.5 cut-off.**
- **Mistake A: dropping "outliers" from the test set.** An isolation forest removes 44% of the test customers. On this data it doesn't flatter the scores: every metric goes slightly down. Its real cost is that the test set no longer describes the customers you will actually have to score.
- **Mistake B: balancing the classes before the split.** Buyers are duplicated until the data is 50/50, then split. Two things go wrong:
  - The test set now has a 50% buy rate, so precision and PR AUC rise mechanically. A model trained on balanced data also pushes its scores up, so recall at 0.5 jumps from 0.003 to 0.936.
  - Copies of the same customer land on both sides of the split. ROC AUC, which ignores the buy rate, only moves from 0.858 to 0.863. That small gap is the actual leak.

  The rest is a test set that no longer looks like the customers you will call.

## 2. The stream

```
data/test.csv ──► Kafka topic ──► Spark Structured Streaming ──► parquet
   (Spark writes        (one JSON message     (micro-batches of 10,000,
    one message per      per customer)         features() + the saved model)
    customer)
```

`stream.py` replays the held-out customers into Kafka. A streaming query reads them in micro-batches and scores each batch with `lab.features` and the model `lab.py` saved. It then checks two things against scoring the same customers in one batch:

- every customer is scored exactly once;
- every score matches the batch score.

Result: **114,333 customers streamed through Kafka; max |stream − batch| = 0.0.** Every score is bit-identical.

Train/serve skew is the usual way a model that tested well goes wrong in production. Here it is ruled out by construction: one `features()` function, one image, one model file. The check makes sure it stays that way.

## Run it

Needs Docker and about 3 GB of free memory.

```
curl -L --create-dirs -o data/raw.zip https://www.kaggle.com/api/v1/datasets/download/anmolkumar/health-insurance-cross-sell-prediction
docker compose run --rm lab python3 lab.py        # ~1 min: the table above, then saves the model
docker compose run --rm lab spark-submit stream.py  # ~3 min: Kafka -> Spark -> parity check
```

Both scripts end in `assert`s. If either finishes, its claims held on your machine.

## Files

| | |
|---|---|
| `lab.py` | features, model, the three evaluations |
| `stream.py` | Kafka producer, streaming scorer, parity check |
| `Dockerfile` | Spark 4.1.3 + Python 3.10; training and streaming run in the same image |
| `docker-compose.yml` | single-node Kafka (KRaft) + the lab container |
