"""A proper MLflow DatasetSource for lakeFS URIs: lakefs://<repo>/<ref>/<path>.

Importing this module registers the source, so `mlflow.data.from_pandas(...,
source="lakefs://repo/commit-or-branch/path/")` resolves with source_type
"lakefs" instead of raising "Could not find a source information resolver".
"""
import os
import tempfile
from typing import Any
from urllib.parse import urlparse

from mlflow.data.dataset_source_registry import register_dataset_source
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource


class LakeFSDatasetSource(FileSystemDatasetSource):
    def __init__(self, uri: str, endpoint: str | None = None):
        self._uri = uri
        self._endpoint = endpoint or os.environ.get("LAKEFS_ENDPOINT", "http://localhost:8000")

    @property
    def uri(self) -> str:
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return "lakefs"

    # --- resolution: how the registry decides this class owns a raw source ---
    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        return isinstance(raw_source, str) and urlparse(raw_source).scheme == "lakefs"

    @classmethod
    def _resolve(cls, raw_source: Any) -> "LakeFSDatasetSource":
        return cls(raw_source)

    # --- (de)serialization: what gets stored on the MLflow run ---
    # The endpoint is stored too: a lakefs:// URI alone doesn't say where the
    # server lives, and consumers (e.g. the UI) need it to build browse links.
    def to_dict(self) -> dict[str, Any]:
        return {"uri": self._uri, "endpoint": self._endpoint}

    @classmethod
    def from_dict(cls, source_dict: dict[str, Any]) -> "LakeFSDatasetSource":
        return cls(source_dict["uri"], endpoint=source_dict.get("endpoint"))

    # --- the part s3:// prefixes can't give you: re-materialize the exact data ---
    def load(self, dst_path: str | None = None) -> str:
        """Download every object under the source URI via the lakeFS S3 gateway.

        With a commit ID in the URI, this returns the same bytes forever.
        """
        import boto3

        parsed = urlparse(self._uri)
        repo = parsed.netloc
        ref, _, path = parsed.path.lstrip("/").partition("/")

        s3 = boto3.client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["LAKEFS_SECRET_ACCESS_KEY"],
        )
        dst_path = dst_path or tempfile.mkdtemp(prefix="lakefs-dataset-")
        prefix = f"{ref}/{path}"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=repo, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix):].lstrip("/") or os.path.basename(obj["Key"])
                local = os.path.join(dst_path, rel)
                os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
                s3.download_file(repo, obj["Key"], local)
        return dst_path


register_dataset_source(LakeFSDatasetSource)
