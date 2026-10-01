# cross-sell-lab

An insurer sells health insurance and wants to sell car insurance to the same customers. Given what it knows about each customer (age, gender, region, how they were sold, their vehicle, their premium), **which customers should it call?**

This repo does two things:

1. **Trains and evaluates a model** that ranks customers by how likely they are to buy. It reports the honest result alongside two common evaluation mistakes, each with the numbers it produces.
2. **Scores customers as a stream.** It sends 114,333 customers through Kafka one message at a time, scores them with Spark Structured Streaming, and checks that every streamed score is identical to scoring the same customers in one batch.

Everything below was produced by the two scripts in this repo, run in the Docker image defined here.

---

## 1. Data

Kaggle, [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) (GPL-2, not redistributed here). Only `train.csv` is used, because Kaggle's `test.csv` has no answers.

| | |
|---|---|
| Rows | 381,109 customers, one row each |
| Columns | 12: an id, 10 customer attributes, and the answer `Response` |
| Missing values | none |
| Buyers (`Response = 1`) | 46,710, which is **12.3%** |

### Features: what the model sees

`features()` in `lab.py` turns each raw row into 10 numbers. It's a plain pandas function with no fitted state, which matters for the stream (section 4).

| Raw column | Feature | Encoding | Range in data |
|---|---|---|---|
| Gender | `male` | 1 if Male, else 0 | |
| Age | `age` | as is | 20 to 85 |
| Driving_License | `licence` | 0/1 as is | |
| Region_Code | `region` | **categorical** | 53 regions |
| Previously_Insured | `insured` | 0/1 as is (already has car insurance) | |
| Vehicle_Age | `vehicle_age` | `< 1 Year`=0, `1-2 Year`=1, `> 2 Years`=2 | |
| Vehicle_Damage | `damage` | 1 if Yes, else 0 | |
| Annual_Premium | `premium` | as is | 2,630 to 540,165 (median 31,669) |
| Policy_Sales_Channel | `channel` | **categorical** | 155 channels |
| Vintage | `vintage` | as is (days as a customer) | 10 to 299 |

`id` is dropped and `Response` is the label. Region and sales channel are stored as numbers, but they are codes: channel 152 is not "more" than channel 26. They are therefore passed to the model as categories, not quantities.

### Split

70% train / 30% test, **stratified** (both sides keep the 12.3% buy rate), seed 0, done once:

| | customers | buyers |
|---|---|---|
| train | 266,776 | 32,697 |
| test | 114,333 | 14,013 |

The test set is used only for the final numbers. Nothing is tuned on it.

---

## 2. The model, and why this one

**Model:** scikit-learn's `HistGradientBoostingClassifier`, which is gradient-boosted decision trees.

**Why gradient-boosted trees:**

- **The data is a table of mixed columns:** binary flags, an ordinal (vehicle age), skewed amounts (premium) and codes. Boosted trees are the standard first choice for this. They pick up non-linear effects and interactions (for example, customers who already have car insurance almost never buy) without hand-built features.
- **No scaling is needed.** Trees split on thresholds, so a premium of 540,165 next to an age of 20 is not a problem.
- **Native categorical splits.** Region (53 values) and channel (155) are split as categories directly. The alternatives are worse: one-hot encoding adds 208 columns, and treating the codes as numbers would be wrong.
- **Size.** 266,776 rows × 10 columns fits in memory. The histogram method buckets each feature into at most 255 bins. All of `lab.py`, which is two model fits plus the isolation forest, runs in about a minute on 4 CPUs. Training does not need Spark.

**Why scikit-learn's version over LightGBM or XGBoost:** it is the same family of algorithm (scikit-learn's version is modelled on LightGBM), and it comes with scikit-learn, so there is no extra dependency. The saved model is one `joblib` file that loads unchanged inside Spark's Python workers.

**Settings:** all scikit-learn defaults, with `random_state=0`:

| parameter | value |
|---|---|
| learning rate | 0.1 |
| max trees (`max_iter`) | 100 |
| max leaves per tree | 31 |
| min customers per leaf | 20 |
| L2 regularisation | 0 |
| early stopping | on (automatic above 10,000 rows): 10% of the training rows are held out, and training stops after 10 trees without improvement in log-loss |
| **trees actually built** | **61** (early stopping halted it) |

**No class rebalancing.** The model is trained on the real 12.3% buy rate, so its scores sit on that scale. The consequence is in section 3: very few customers score above 0.5.

**What was *not* done, plainly:**
- no hyperparameter tuning;
- no cross-validation (one split);
- no comparison against another model (no logistic-regression baseline);
- no calibration check;
- no threshold tuning.

So this README shows how a sensible default model performs and how evaluation mistakes distort it. It does not claim this is the best model for the data.

---

## 3. Results: the honest evaluation and two mistakes

`lab.py` scores the same kind of model three ways and prints this table:

| protocol | rows | buy rate | ROC AUC | PR AUC | precision@0.5 | recall@0.5 | precision, top 10% |
|---|---|---|---|---|---|---|---|
| honest | 114,333 | 0.123 | 0.858 | 0.371 | 0.471 | 0.003 | 0.398 |
| A: outliers dropped from test | 63,862 | 0.143 | 0.838 | 0.370 | 0.432 | 0.002 | 0.393 |
| B: oversampled before split | 200,640 | 0.500 | 0.863 | 0.808 | 0.743 | 0.936 | 0.850 |

**What each column means:**

- **ROC AUC**: the chance that a random buyer is ranked above a random non-buyer. It doesn't depend on the buy rate.
- **PR AUC**: average precision across all cut-offs. It *does* depend on the buy rate.
- **precision@0.5 / recall@0.5**: call everyone the model gives at least 50%. Precision is the share of those calls that buy; recall is the share of all buyers you reach.
- **precision, top 10%**: call the 10% of customers the model ranks highest. This is the share of those calls that buy.

### Honest

- **The ranking is good:** ROC AUC 0.858.
- **The 0.5 cut-off is useless here.** Only **87** of 114,333 test customers score 0.5 or more. 41 of them buy, which reaches 41 of the 14,013 buyers (recall 0.003). With a 12% buy rate, few customers are ever more likely than not to buy.
- **Use the ranking instead.** Calling the top 10% means **11,433 calls**, everyone scoring at least 0.352. **4,555 of them buy (39.8%)**, against 12.3% for random calls, which is **3.2×**. Those 10% of calls reach **32.5% of all buyers**.

### Mistake A: dropping "outliers" from the test set

An isolation forest (100 trees, default settings), fitted on the training features, flags **50,471 of the 114,333 test customers (44.1%)** as outliers. Those include 4,905 buyers. Dropping them leaves 63,862 customers.

On this data the mistake **does not flatter** the model: ROC AUC falls from 0.858 to 0.838. The dropped customers bought at 9.7%, so the buy rate of those left rises to 14.3%. The real damage is that the test set no longer represents the customers you'll actually have to score. You can't drop real customers at prediction time.

### Mistake B: balancing the classes before the split

To reach 50/50, **287,689** randomly duplicated buyer rows are added to the full data (giving 668,798 rows), *then* it is split 70/30 into 468,158 training and 200,640 test rows. Because the duplication happened before the split, **99,767 test rows (49.7%) are customers who also appear in the training set, and so do 99.4% of the test buyers.** The model is being graded on people it trained on.

What that does to the numbers:

- **ROC AUC barely moves:** 0.858 → 0.863. That is likely because the trees can't memorise individuals: every leaf needs at least 20 training rows. So seeing a customer twice helps only slightly.
- **Everything that depends on the buy rate jumps:**
  - PR AUC 0.371 → 0.808
  - precision@0.5 0.471 → 0.743
  - recall@0.5 0.003 → **0.936**

  The test buy rate went from 12% to 50%, and a model trained on 50/50 data scores everyone higher.

Reported this way, the model looks as if it finds 94% of buyers at 74% precision. In reality it finds 0.3% at that cut-off.

`lab.py` ends with an `assert` that mistake B inflates every score column. Mistake A has no assert, because it doesn't inflate them on this data.

---

## 4. The stream

```
data/test.csv ─► Spark batch write ─► Kafka topic ─► Spark Structured Streaming ─► parquet ─► parity check
 114,333 rows    1 JSON msg / row     1 partition    12 micro-batches              12 files    vs batch scoring
```

**Why stream at all:** in production, customers arrive over time and get scored as they arrive. The usual way a model that tested well fails there is **train/serve skew**: the live system computes features slightly differently from training. This half of the repo replays the test customers through a real broker and a real streaming engine, then proves the scores match.

**Spark isn't needed at this size.** 114k rows fit in pandas. It's here because the pattern carries over unchanged to volumes that need it: Kafka source, micro-batches, checkpoints and Python scoring inside Spark.

**Step by step (`stream.py`):**

1. **Produce.** A Spark batch job turns each of the 114,333 test customers into one compact JSON message (228 bytes on average, about 26 MB in total). It writes them to a fresh Kafka topic, `customers_<unix time>`. A new topic per run means a rerun can't double-count.
2. **Broker.** `apache/kafka:4.3.1`, a single broker in KRaft mode (no ZooKeeper), heap capped at 256 MB. The topic is auto-created with **1 partition**.
3. **Consume.** A Spark Structured Streaming query (Spark 4.1.3, local mode, 4 cores):
   - reads the topic from the earliest offset;
   - takes at most 10,000 messages per micro-batch, giving **12 micro-batches** (11 × 10,000 + 1 × 4,333);
   - uses `trigger(availableNow=True)`: process everything currently in the topic, then stop. It's a replay, not an always-on job.
4. **Score.** Each micro-batch is parsed with the original table's schema. `mapInPandas` then hands it to Python workers as Arrow batches, and they run the **same `features()` function and the same model file** as `lab.py`.
5. **Sink.** `(id, score)` rows are appended to parquet, giving 12 files. Kafka offsets and batch commits are checkpointed. That's the standard mechanism that lets a restarted query resume without scoring customers twice. This script doesn't test a restart, because each run uses a fresh topic.
6. **Check.** The script reads the parquet back and scores `data/test.csv` in one batch with the same model. It asserts two things:
   - **every customer is scored exactly once**: 114,333 rows, 114,333 unique ids, the same ids as the test set;
   - **every score matches**: max |stream − batch| < 1e-12.

**Result:**
```
114,333 customers streamed through Kafka; max |stream - batch| = 0.0e+00
```
Every streamed score is **bit-identical** to batch scoring.

**Limits of this test:**
- One machine, one partition. It shows the pipeline is *correct*, not that it *scales*.
- Throughput was not benchmarked. The whole script takes about 3 minutes, including JVM start-up and the first-time download of the Kafka connector.
- The messages carry the `Response` column, because the replay sends the test file as it is. `features()` never reads it.

---

## Run it

Needs Docker, and about 3 GB of free memory for Kafka plus Spark.

```
curl -L --create-dirs -o data/raw.zip https://www.kaggle.com/api/v1/datasets/download/anmolkumar/health-insurance-cross-sell-prediction
docker compose run --rm lab python3 lab.py          # ~1 min: section 3's table; saves data/model.joblib and data/test.csv
docker compose run --rm lab spark-submit stream.py  # ~3 min: section 4; needs the model from the line above
```

`docker compose run` starts Kafka automatically. Both scripts end in `assert`s: if a script finishes, its claims held on your machine.

## Environment the numbers come from

| | |
|---|---|
| Image | official `spark:4.1.3-python3` (Ubuntu 22.04, OpenJDK 17.0.20.1) |
| Python | 3.10.12 |
| Libraries | pandas 2.3.3, scikit-learn 1.7.2, numpy 2.2.6, pyarrow 25.0.1 |
| Kafka | `apache/kafka:4.3.1`; connector `spark-sql-kafka-0-10_2.13:4.1.3` |
| Machine | Docker VM with 4 CPUs, 3.8 GB RAM |

Training and streaming run in the **same image**, so the model file is always loaded by the same Python and scikit-learn versions that wrote it.

## Files

| file | what's in it |
|---|---|
| `lab.py` | `features()`, the model, the three evaluations, the table, saving the model |
| `stream.py` | Kafka producer, streaming query, scoring, parity check |
| `Dockerfile` | the Spark image plus the three pinned Python libraries and the Kafka connector setting |
| `docker-compose.yml` | the Kafka broker and the lab container (which shares Kafka's network, so the broker is `localhost:9092`) |
| `requirements.txt` | pinned pandas, scikit-learn, pyarrow |
