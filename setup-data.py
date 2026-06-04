import io
import os

import boto3
import requests
import yaml
import pandas as pd
from botocore.config import Config
from sklearn.datasets import make_classification

# lakeFS connection: credentials + endpoint come from the lakectl config
# (override any of them with LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY_ID /
# LAKEFS_SECRET_ACCESS_KEY env vars).
LAKECTL_CONFIG = os.environ.get(
    "LAKECTL_CONFIG",
    os.path.expanduser("~/work/configs/lakectl/enterprise_hackathon_2026.yaml"),
)
with open(LAKECTL_CONFIG) as f:
    _cfg = yaml.safe_load(f)

# The lakectl endpoint includes /api/v1; the S3 gateway is the bare host.
API_ENDPOINT = os.environ.get("LAKEFS_API_ENDPOINT", _cfg["server"]["endpoint_url"]).rstrip("/")
GATEWAY_ENDPOINT = os.environ.get("LAKEFS_ENDPOINT", API_ENDPOINT.removesuffix("/api/v1"))
ACCESS_KEY = os.environ.get("LAKEFS_ACCESS_KEY_ID", _cfg["credentials"]["access_key_id"])
SECRET_KEY = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", _cfg["credentials"]["secret_access_key"])

REPO, BRANCH, PREFIX = "ml-demo", "main", "data/churn/"
# lakeFS S3 gateway maps bucket=<repo>, key=<ref>/<path>.
KEY = f"{BRANCH}/{PREFIX}churn.csv"

# Write THROUGH the lakeFS S3 gateway (path-style addressing) so the object is
# versioned by lakeFS — not straight into the underlying bucket.
s3 = boto3.client(
    "s3",
    endpoint_url=GATEWAY_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(s3={"addressing_style": "path"}),
)

# fabricate something that looks like a real tabular dataset
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=42)
df = pd.DataFrame(X, columns=["tenure", "monthly_charge", "num_logins",
                              "support_tickets", "data_usage"])
df["churned"] = y

# upload to the lakeFS branch as CSV
buf = io.StringIO()
df.to_csv(buf, index=False)
s3.put_object(Bucket=REPO, Key=KEY, Body=buf.getvalue())
print(f"Uploaded {len(df)} rows to lakefs://{REPO}/{KEY}")

# commit on the branch so the data sits at an immutable commit, which is what
# compute.py reads (it resolves branch -> head commit before loading).
resp = requests.post(
    f"{API_ENDPOINT}/repositories/{REPO}/branches/{BRANCH}/commits",
    json={"message": "seed churn data"},
    auth=(ACCESS_KEY, SECRET_KEY),
)
resp.raise_for_status()
print("Committed:", resp.json().get("id"))