import io, os
import boto3
import requests
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import lakefs_source  # noqa: F401 -- registers lakefs:// as an MLflow dataset source
import lakefs_artifact_repo  # noqa: F401 -- registers lakefs:// as an MLflow artifact store

# Credentials MLflow uses IF your server pushes artifacts to MinIO.
# Harmless if your server stores artifacts locally (see note at the end).
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

# lakeFS: API + S3 gateway on :8000 -- credentials come from ~/.lakectl.yaml
# (or override with LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY_ID / LAKEFS_SECRET_ACCESS_KEY env vars)
import yaml
with open(os.path.expanduser("~/.lakectl.yaml")) as f:
    _lakectl = yaml.safe_load(f)
LAKEFS_ENDPOINT = os.environ.setdefault("LAKEFS_ENDPOINT", "http://localhost:8000")
os.environ.setdefault("LAKEFS_ACCESS_KEY_ID", _lakectl["credentials"]["access_key_id"])
os.environ.setdefault("LAKEFS_SECRET_ACCESS_KEY", _lakectl["credentials"]["secret_access_key"])

REPO, BRANCH, PREFIX = "ml-demo", "main", "data/churn/"

# --- 1. resolve the branch to a commit, so the run records an IMMUTABLE source ---
COMMIT = requests.get(
    f"{LAKEFS_ENDPOINT}/api/v1/repositories/{REPO}/branches/{BRANCH}",
    auth=(os.environ["LAKEFS_ACCESS_KEY_ID"], os.environ["LAKEFS_SECRET_ACCESS_KEY"]),
).json()["commit_id"]
DATA_URI = f"lakefs://{REPO}/{COMMIT}/{PREFIX}"

# --- 2. read every object under the prefix, through the lakeFS S3 gateway ---
s3 = boto3.client("s3", endpoint_url=LAKEFS_ENDPOINT,
                  aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["LAKEFS_SECRET_ACCESS_KEY"])
keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=REPO, Prefix=f"{COMMIT}/{PREFIX}")
        for o in page.get("Contents", [])]
print(f"Reading {len(keys)} files from {DATA_URI}")
parts = [pd.read_csv(io.BytesIO(s3.get_object(Bucket=REPO, Key=k)["Body"].read())) for k in keys]
df = pd.concat(parts, ignore_index=True)

X, y = df.drop(columns=["churned"]), df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. point at the MLflow server and name the project ---
# Artifact location is fixed at experiment creation: models/artifacts get
# written to the lakeFS BRANCH (mutable), then committed below to pin them.
mlflow.set_tracking_uri("http://localhost:5001")
if mlflow.get_experiment_by_name("churn-demo") is None:
    mlflow.create_experiment("churn-demo", artifact_location=f"lakefs://{REPO}/{BRANCH}/mlflow")
mlflow.set_experiment("churn-demo")

# --- 3. one attempt = one run. Change these between runs. ---
n_estimators, max_depth = 100, 5

with mlflow.start_run(run_name=f"rf-depth-{max_depth}"):
    # record WHICH data fed this run (the data-lineage touch)
    # NB: the tracking server dedupes datasets by (name, digest) and keeps the
    # FIRST source it saw -- so put the commit in the name to get a fresh record.
    # point the source at the CSV file (not the prefix) so the lakeFS URI links
    # straight to the object viewer and its DuckDB query panel.
    dataset = mlflow.data.from_pandas(df, source=f"{DATA_URI}churn.csv", name=f"churn@{COMMIT[:8]}", targets="churned")
    mlflow.log_input(dataset, context="training")

    # the settings we chose
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # the (trivial) training
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # the results we got
    preds = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1", f1_score(y_test, preds))

    # the model artifact itself -- uploaded to the lakeFS branch via the gateway
    model_info = mlflow.sklearn.log_model(model, name="model")
    print("Model stored at:", model_info.artifact_path or mlflow.get_logged_model(model_info.model_id).artifact_location)

    print("Logged run:", mlflow.active_run().info.run_id)

# --- 4. pin it: commit the uploaded model so the artifacts are immutable too ---
commit = requests.post(
    f"{LAKEFS_ENDPOINT}/api/v1/repositories/{REPO}/branches/{BRANCH}/commits",
    json={"message": f"model for run {mlflow.last_active_run().info.run_id}"},
    auth=(os.environ["LAKEFS_ACCESS_KEY_ID"], os.environ["LAKEFS_SECRET_ACCESS_KEY"]),
).json()
print("Model pinned in lakeFS commit:", commit.get("id", commit))