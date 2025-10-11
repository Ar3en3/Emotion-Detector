import os, time, json, boto3
from decimal import Decimal

# Force a session that uses your profile & region even if envs are missing
region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2"
profile = os.environ.get("AWS_PROFILE") or "emotion-admin"
session = boto3.Session(profile_name=profile, region_name=region)

print("Using profile:", profile)
print("Region:", region)

# ---- S3 smoke test ----
s3 = session.client("s3")
bucket = "emotion-app-729798775712-ap-southeast-2"
key = f"smoke/s3-ok-{int(time.time())}.txt"
s3.put_object(Bucket=bucket, Key=key, Body=b"hello from the app env")
print("S3 OK ->", f"s3://{bucket}/{key}")

# ---- DynamoDB smoke test ----
ddb = session.resource("dynamodb")
table = ddb.Table("emotion_logs")

# Use Decimal for any non-integer numbers
item = {
    "id": f"smoke-{int(time.time())}",
    "ts": int(time.time()),
    "emotion": "happy",
    "confidence": Decimal("0.99"),
    "source": "smoke-test"
}
table.put_item(Item=item)
print("DynamoDB OK -> emotion_logs", json.dumps(item, default=str))
