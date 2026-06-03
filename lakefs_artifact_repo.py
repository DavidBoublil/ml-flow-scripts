"""An MLflow ArtifactRepository for lakefs:// URIs — the write-side twin of
LakeFSDatasetSource.

Importing this module registers the repository, so an experiment created with
artifact_location="lakefs://repo/branch/path" makes every run/model artifact
upload go through the lakeFS S3 gateway (bucket = repo, key = branch/path/...).

NB: write to a BRANCH, never a commit ID — commits are immutable. Commit after
the run to pin the uploaded artifacts.
"""
import os
import urllib.parse

import boto3
from mlflow.store.artifact.artifact_repository_registry import _artifact_repository_registry
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository


class LakeFSArtifactRepository(S3ArtifactRepository):
    def parse_s3_compliant_uri(self, uri):
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "lakefs":
            raise Exception(f"Not a lakeFS URI: {uri}")
        # lakefs://<repo>/<ref>/<path> -> gateway bucket=<repo>, key=<ref>/<path>
        return parsed.netloc, parsed.path.lstrip("/")

    def _get_s3_client(self):
        return boto3.client(
            "s3",
            endpoint_url=os.environ.get("LAKEFS_ENDPOINT", "http://localhost:8000"),
            aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["LAKEFS_SECRET_ACCESS_KEY"],
        )


_artifact_repository_registry.register("lakefs", LakeFSArtifactRepository)
