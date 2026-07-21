import logging
from typing import Iterable

import requests


class ChromaStore:
    def __init__(
        self,
        base_url,
        tenant,
        database,
        collection_name,
        timeout_seconds=15,
        token="",
    ):
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.database = database
        self.collection_name = collection_name
        self.timeout_seconds = timeout_seconds
        self.token = token
        self._collection_id = None

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["x-chroma-token"] = self.token
        return headers

    def _request(self, method, path, expected_statuses=(200,), payload=None):
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )

        if response.status_code not in expected_statuses:
            detail = response.text.strip() or response.reason
            raise RuntimeError(f"ChromaDB 요청 실패 ({response.status_code}): {detail}")

        if not response.text:
            return None
        return response.json()

    def _collections_path(self):
        return f"/api/v2/tenants/{self.tenant}/databases/{self.database}/collections"

    def ensure_collection(self):
        if self._collection_id:
            return self._collection_id

        collections = self._request("GET", self._collections_path())
        for collection in collections:
            if collection.get("name") == self.collection_name:
                self._collection_id = collection["id"]
                return self._collection_id

        created = self._request(
            "POST",
            self._collections_path(),
            payload={
                "name": self.collection_name,
                "get_or_create": True,
                "metadata": {"hnsw:space": "cosine"},
            },
        )
        self._collection_id = created["id"]
        return self._collection_id

    def invalidate_collection_cache(self):
        self._collection_id = None

    def health_check(self):
        try:
            collection_id = self.ensure_collection()
            return {
                "status": "ok",
                "error": "",
                "collection_id": collection_id,
            }
        except Exception as exc:
            logging.warning("Chroma health check failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "collection_id": None,
            }

    def collection_records_path(self, action):
        collection_id = self.ensure_collection()
        return (
            f"/api/v2/tenants/{self.tenant}/databases/{self.database}"
            f"/collections/{collection_id}/{action}"
        )

    def upsert_records(self, records):
        if not records:
            return

        payload = {
            "ids": [record["id"] for record in records],
            "embeddings": [record["embedding"] for record in records],
            "documents": [record["document"] for record in records],
            "metadatas": [record["metadata"] for record in records],
        }
        self._request(
            "POST",
            self.collection_records_path("upsert"),
            expected_statuses=(200, 201),
            payload=payload,
        )

    def query(self, query_embedding, n_results, where=None):
        payload = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            payload["where"] = where

        return self._request("POST", self.collection_records_path("query"), payload=payload)

    def delete_by_where(self, where):
        self._request(
            "POST",
            self.collection_records_path("delete"),
            expected_statuses=(200, 201),
            payload={"where": where},
        )

    def delete_by_ids(self, ids: Iterable[str]):
        ids = list(ids)
        if not ids:
            return

        self._request(
            "POST",
            self.collection_records_path("delete"),
            expected_statuses=(200, 201),
            payload={"ids": ids},
        )

    def get_records(self, where=None, include=None):
        payload = {"include": include or ["metadatas"]}
        if where:
            payload["where"] = where
        return self._request("POST", self.collection_records_path("get"), payload=payload)

    def clear_collection(self):
        collection_id = self.ensure_collection()
        self._request(
            "DELETE",
            f"/api/v2/tenants/{self.tenant}/databases/{self.database}/collections/{collection_id}",
        )
        self.invalidate_collection_cache()
        logging.info("Cleared Chroma collection: %s", self.collection_name)
