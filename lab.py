"""Which car-insurance customers to call: one honest evaluation, then two common mistakes that flatter it."""
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 0


def features(df):
    """Raw customer rows -> model inputs. Stateless, so the batch job and the stream compute the same thing."""
    return pd.DataFrame({
        "male": (df["Gender"] == "Male").astype(int),
        "age": df["Age"],
        "licence": df["Driving_License"],
        "region": df["Region_Code"].astype(int),
        "insured": df["Previously_Insured"],
        "vehicle_age": df["Vehicle_Age"].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}),
        "damage": (df["Vehicle_Damage"] == "Yes").astype(int),
        "premium": df["Annual_Premium"],
        "channel": df["Policy_Sales_Channel"].astype(int),
        "vintage": df["Vintage"],
    })


def fit(train):
    # region and channel are codes, not quantities
    model = HistGradientBoostingClassifier(categorical_features=["region", "channel"], random_state=SEED)
    return model.fit(features(train), train["Response"])


def evaluate(model, test):
    y = test["Response"].to_numpy()
    p = model.predict_proba(features(test))[:, 1]
    return {
        "rows": len(test),
        "buy rate": y.mean(),
        "ROC AUC": roc_auc_score(y, p),
        "PR AUC": average_precision_score(y, p),
        "precision@0.5": precision_score(y, p >= 0.5, zero_division=0),
        "recall@0.5": recall_score(y, p >= 0.5),
        "precision, top 10%": y[np.argsort(p)[-len(p) // 10:]].mean(),
    }


if __name__ == "__main__":
    df = pd.read_csv(zipfile.ZipFile("data/raw.zip").open("train.csv"))
    train, test = train_test_split(df, test_size=0.3, stratify=df["Response"], random_state=SEED)
    model = fit(train)

    # Mistake A: drop "outliers" from the test set. A real customer can't be dropped at scoring time.
    iso = IsolationForest(random_state=SEED).fit(features(train))
    filtered = test[iso.predict(features(test)) == 1]

    # Mistake B: balance the classes by duplicating buyers before the split, so copies land on both sides.
    buyers = df[df["Response"] == 1]
    balanced = pd.concat([df, buyers.sample(len(df) - 2 * len(buyers), replace=True, random_state=SEED)])
    train_b, test_b = train_test_split(balanced, test_size=0.3, stratify=balanced["Response"], random_state=SEED)

    rows = {
        "honest": evaluate(model, test),
        "A: outliers dropped from test": evaluate(model, filtered),
        "B: oversampled before split": evaluate(fit(train_b), test_b),
    }
    cols = list(rows["honest"])
    print("| protocol | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|")
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{v:,}" if isinstance(v, int) else f"{v:.3f}" for v in r.values()) + " |")

    # Mistake A is not asserted: on this data it shrinks the test set without flattering the scores.
    honest, b = rows["honest"], rows["B: oversampled before split"]
    assert all(b[k] > honest[k] for k in cols[2:]), "oversampling before the split should flatter every score"

    joblib.dump(model, "data/model.joblib")
    test.to_csv("data/test.csv", index=False)
