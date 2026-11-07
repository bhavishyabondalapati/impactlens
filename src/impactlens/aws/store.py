"""Where a run's files live: an S3 bucket on AWS, a local folder in tests.

Every Lambda starts with an empty /tmp, so each step downloads what it needs from the bucket
and uploads what it made. Both stores use the same keys, e.g. ``layers/<run_id>/flooded.tif``.

Bucket layout (one private bucket):
  requests/<run_id>.json      the request (AOI + date windows); uploading it starts a run
  models/                     floods.onnx, oscd.onnx (+ metrics.json with tuned thresholds)
  index/                      sops.faiss, chunks.jsonl, embedder/minilm.onnx + tokenizer.json
  osm/                        Overpass answers, cached per AOI (fetched once per AOI)
  layers/<run_id>/            scene, GeoTIFF layers, geo outputs, fact sheet (expire: 30 days)
  reports/<run_id>/           the reports + run.json; work/ holds drafts during the run
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class LocalStore:
    """A folder that behaves like the bucket (tests and local dry runs)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def download(self, key: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / key, path)
        return path

    def upload(self, path: str | Path, key: str) -> None:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    def read_text(self, key: str) -> str:
        return (self.root / key).read_text()

    def write_text(self, key: str, text: str) -> None:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    def write_json(self, key: str, obj) -> None:
        self.write_text(key, json.dumps(obj, indent=1, ensure_ascii=False))


class S3Store(LocalStore):
    """The same methods on S3 (boto3 is preinstalled in the Lambda Python runtime)."""

    def __init__(self, bucket: str, client=None):
        import boto3

        self.bucket = bucket
        self.s3 = client or boto3.client("s3")

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as err:
            if err.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def download(self, key: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(path))
        return path

    def upload(self, path: str | Path, key: str) -> None:
        self.s3.upload_file(str(path), self.bucket, key)

    def read_text(self, key: str) -> str:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode()

    def write_text(self, key: str, text: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode(),
                           ContentType="application/json" if key.endswith(".json")
                           else "text/plain; charset=utf-8")  # fmt: skip
