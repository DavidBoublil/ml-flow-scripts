import io
import pandas as pd
import boto3
from sklearn.datasets import make_classification

BUCKET, KEY = "ml-demo", "data/churn.csv"

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

# create the bucket if it doesn't exist yet
if BUCKET not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
    s3.create_bucket(Bucket=BUCKET)

# fabricate something that looks like a real tabular dataset
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=42)
df = pd.DataFrame(X, columns=["tenure", "monthly_charge", "num_logins",
                              "support_tickets", "data_usage"])
df["churned"] = y

# upload straight to MinIO as CSV
buf = io.StringIO()
df.to_csv(buf, index=False)
s3.put_object(Bucket=BUCKET, Key=KEY, Body=buf.getvalue())
print(f"Uploaded {len(df)} rows to s3://{BUCKET}/{KEY}")