"""Which car-insurance customers to call: one honest evaluation, a baseline, then two common mistakes that flatter it.
Every number in the README is printed by this script or by stream.py."""
import platform
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def score(model, df):
    return model.predict_proba(features(df))[:, 1]


def ranked(p):
    """Positions of the customers, highest score first: the order you would call them in."""
    return np.argsort(p)[::-1]


def evaluate(y, p):
    return {
        "rows": len(y),
        "buy rate": y.mean(),
        "ROC AUC": roc_auc_score(y, p),
        "PR AUC": average_precision_score(y, p),
        "precision@0.5": precision_score(y, p >= 0.5, zero_division=0),
        "recall@0.5": recall_score(y, p >= 0.5),
        "precision, top 10%": y[ranked(p)[:len(p) // 10]].mean(),
    }


if __name__ == "__main__":
    print(f"env: python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, scikit-learn {sklearn.__version__}")
    df = pd.read_csv(zipfile.ZipFile("data/raw.zip").open("train.csv"))
    print(f"data: {len(df):,} customers, {df.shape[1]} columns, {df.isna().sum().sum()} missing values, "
          f"{df.Response.sum():,} buyers ({df.Response.mean():.1%})")
    print(f"ranges: age {df.Age.min()}-{df.Age.max()}, premium {df.Annual_Premium.min():,.0f}-{df.Annual_Premium.max():,.0f} "
          f"(median {df.Annual_Premium.median():,.0f}), vintage {df.Vintage.min()}-{df.Vintage.max()} days, "
          f"{df.Region_Code.nunique()} regions, {df.Policy_Sales_Channel.nunique()} sales channels")
    for insured, g in df.groupby("Previously_Insured")["Response"]:
        print(f"previously insured = {insured}: {g.sum():,} of {len(g):,} bought ({g.mean():.2%})")

    train, test = train_test_split(df, test_size=0.3, stratify=df["Response"], random_state=SEED)
    y = test["Response"].to_numpy()
    print(f"split: train {len(train):,} ({train.Response.sum():,} buyers), test {len(test):,} ({y.sum():,} buyers)")

    model = fit(train)
    p = score(model, test)
    print(f"model: {model.n_iter_} trees built before early stopping; " + ", ".join(
        f"{k} {getattr(model, k)}" for k in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                                             "l2_regularization", "early_stopping", "validation_fraction", "n_iter_no_change")))

    # Baseline: a linear model on the same features, codes one-hot encoded, everything else scaled.
    baseline = make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["region", "channel"]), remainder=StandardScaler()),
        LogisticRegression(max_iter=1000),
    ).fit(features(train), train["Response"])
    pb = score(baseline, test)
    print(f"baseline: {len(baseline[0].get_feature_names_out())} input columns after encoding")

    # Mistake A: drop "outliers" from the test set. A real customer can't be dropped at scoring time.
    keep = IsolationForest(random_state=SEED).fit(features(train)).predict(features(test)) == 1
    print(f"A: {(~keep).sum():,} of {len(y):,} test customers flagged as outliers and dropped ({(~keep).mean():.1%}), "
          f"{y[~keep].sum():,} of them buyers ({y[~keep].mean():.1%} buy rate)")

    # Mistake B: balance the classes by duplicating buyers before the split, so copies land on both sides.
    buyers = df[df["Response"] == 1]
    dupes = buyers.sample(len(df) - 2 * len(buyers), replace=True, random_state=SEED)
    balanced = pd.concat([df, dupes])
    train_b, test_b = train_test_split(balanced, test_size=0.3, stratify=balanced["Response"], random_state=SEED)
    seen = test_b["id"].isin(train_b["id"])
    print(f"B: {len(dupes):,} duplicated buyer rows added -> {len(balanced):,} rows; train {len(train_b):,}, test {len(test_b):,}")
    print(f"B: {seen.sum():,} test rows ({seen.mean():.1%}) are customers who are also in training, "
          f"including {seen[test_b.Response == 1].mean():.1%} of the test buyers")

    rows = {
        "honest": evaluate(y, p),
        "baseline: logistic regression": evaluate(y, pb),
        "A: outliers dropped from test": evaluate(y[keep], p[keep]),
        "B: oversampled before split": evaluate(test_b["Response"].to_numpy(), score(fit(train_b), test_b)),
    }
    cols = list(rows["honest"])
    print("| protocol | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|")
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{v:,}" if isinstance(v, int) else f"{v:.3f}" for v in r.values()) + " |")

    print(f"boosted: {(p >= 0.5).sum()} customers score >= 0.5, {y[p >= 0.5].sum()} of them buy")
    for name, s in (("boosted", p), ("baseline", pb)):
        hits = np.cumsum(y[ranked(s)])  # buyers reached after each call, best-ranked customer first
        k = len(s) // 10
        print(f"{name}: top 10% = {k:,} calls (score >= {s[ranked(s)[k - 1]]:.3f}): {hits[k - 1]:,} buy "
              f"({hits[k - 1] / k:.1%}, {hits[k - 1] / k / y.mean():.1f}x random), reaching {hits[k - 1] / y.sum():.1%} of all buyers")
        print(f"{name}: buyers reached calling the top 10%, 20%, ..., 100%: "
              + ", ".join(f"{hits[len(s) * d // 10 - 1] / y.sum():.1%}" for d in range(1, 11)))

    # Mistake A is not asserted: on this data it shrinks the test set without flattering the scores.
    honest, b = rows["honest"], rows["B: oversampled before split"]
    assert all(b[k] > honest[k] for k in cols[2:]), "oversampling before the split should flatter every score"

    joblib.dump(model, "data/model.joblib")
    test.to_csv("data/test.csv", index=False)
