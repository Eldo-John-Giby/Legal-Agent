from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    text: str


@dataclass(frozen=True)
class CaseSummary:
    title: str | None
    jurisdiction: str | None
    plaintiff: str | None
    defendant: str | None
    claims: List[str]
    defenses: List[str]
    evidence_ids: List[str]


_EVIDENCE_LINE_RE = re.compile(r"^(E\d+)\s*:\s*(.+)\s*$")


def parse_case_text(case_text: str) -> tuple[CaseSummary, List[EvidenceItem]]:
    title: str | None = None
    jurisdiction: str | None = None
    plaintiff: str | None = None
    defendant: str | None = None
    claims: List[str] = []
    defenses: List[str] = []
    evidence: List[EvidenceItem] = []

    section: str | None = None

    for raw in case_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip() or None
            continue
        if line.upper().startswith("JURISDICTION:"):
            jurisdiction = line.split(":", 1)[1].strip() or None
            continue

        if line.upper() == "PARTIES:":
            section = "parties"
            continue
        if line.upper().startswith("CLAIMS"):
            section = "claims"
            continue
        if line.upper().startswith("DEFENSES"):
            section = "defenses"
            continue
        if line.upper() == "EVIDENCE:":
            section = "evidence"
            continue

        if section == "parties":
            if line.startswith("-") and ":" in line:
                k, v = line[1:].split(":", 1)
                k = k.strip().upper()
                v = v.strip()
                if k == "PLAINTIFF":
                    plaintiff = v
                elif k == "DEFENDANT":
                    defendant = v
            continue

        if section == "claims":
            if line.startswith("-"):
                claims.append(line[1:].strip())
            continue

        if section == "defenses":
            if line.startswith("-"):
                defenses.append(line[1:].strip())
            continue

        if section == "evidence":
            m = _EVIDENCE_LINE_RE.match(line)
            if m:
                evidence.append(EvidenceItem(evidence_id=m.group(1), text=m.group(2)))
            continue

    summary = CaseSummary(
        title=title,
        jurisdiction=jurisdiction,
        plaintiff=plaintiff,
        defendant=defendant,
        claims=claims,
        defenses=defenses,
        evidence_ids=[e.evidence_id for e in evidence],
    )
    return summary, evidence


class EvidenceStore:
    def upsert_all(self, case_id: str, items: Iterable[EvidenceItem], batch_size: int = 40) -> None:
        raise NotImplementedError

    def delete_all(self, case_id: str) -> None:
        raise NotImplementedError

    def search(self, case_id: str, query: str, k: int = 5) -> List[EvidenceItem]:
        raise NotImplementedError

    def get(self, case_id: str, evidence_id: str) -> Optional[EvidenceItem]:
        raise NotImplementedError

    @classmethod
    def get_authority_store(cls) -> Optional[EvidenceStore]:
        """Factory to get the singleton store for legal precedents."""
        return WeaviateEvidenceStore(class_name="AuthorityChunk") if WeaviateEvidenceStore().is_available() else None


def parse_precedents_text(text: str) -> List[EvidenceItem]:
    """Parses data/precedents.txt into EvidenceItems."""
    items = []
    # Split by blocks starting with "ID: "
    blocks = re.split(r"\n\n(?=ID: )", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        # Extract ID from "ID: PRE_ROYAPPA" etc.
        id_line = [l for l in lines if l.startswith("ID: ")]
        if id_line:
            precedent_id = id_line[0].replace("ID:", "").strip()
            items.append(EvidenceItem(evidence_id=precedent_id, text=block.strip()))
    return items


class LocalEvidenceStore(EvidenceStore):
    def __init__(self, items: List[EvidenceItem]):
        self._items = items
        self._by_id = {e.evidence_id: e for e in items}

    def upsert_all(self, case_id: str, items: Iterable[EvidenceItem]) -> None:
        _ = case_id, items

    def delete_all(self, case_id: str) -> None:
        _ = case_id
        self._items = []
        self._by_id = {}

    def get(self, case_id: str, evidence_id: str) -> Optional[EvidenceItem]:
        _ = case_id
        return self._by_id.get(evidence_id)

    def search(self, case_id: str, query: str, k: int = 5) -> List[EvidenceItem]:
        _ = case_id
        q = {t for t in re.split(r"\W+", query.lower()) if t}
        if not q:
            return self._items[:k]

        def score(item: EvidenceItem) -> int:
            toks = {t for t in re.split(r"\W+", item.text.lower()) if t}
            return len(q & toks)

        return sorted(self._items, key=score, reverse=True)[:k]


class WeaviateEvidenceStore(EvidenceStore):
    """Thin HTTP wrapper around Weaviate for evidence RAG.

    This expects your Weaviate instance to have a text vectorizer enabled (e.g., text2vec-transformers).
    If Weaviate is unavailable or misconfigured, you should fall back to LocalEvidenceStore.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        class_name: str = "EvidenceChunk",
        vectorizer: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")).rstrip("/")
        self.class_name = class_name
        # Default to "none" so vanilla Weaviate (no vectorizer module) still works via BM25.
        self.vectorizer = vectorizer or os.getenv("WEAVIATE_VECTORIZER", "none")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        import time
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        max_retries = 3
        
        for attempt in range(max_retries):
            req = Request(
                url=f"{self.base_url}{path}",
                method=method,
                data=data,
                headers=self._headers(),
            )
            try:
                with urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body) if body else {}
            except (HTTPError, URLError, TimeoutError) as e:
                # Retry on network/timeout issues or 5xx server errors
                is_retryable = not isinstance(e, HTTPError) or e.code >= 500
                if is_retryable and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                
                detail = ""
                if isinstance(e, HTTPError):
                    try:
                        detail = e.read().decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                raise RuntimeError(
                    f"Weaviate HTTP {getattr(e, 'code', 'Error')} for {method} {path}: {detail or str(e)}"
                ) from e
            except Exception as e:
                if "Remote end closed connection" in str(e) and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise

    def is_available(self) -> bool:
        try:
            self._request("GET", "/v1/meta")
            return True
        except Exception:
            return False

    def count(self, case_id: str) -> int:
        gql = {
            "query": """
            {
              Aggregate {
                %s(where: {path: [\"case_id\"], operator: Equal, valueText: %s}) {
                  meta { count }
                }
              }
            }
            """
            % (self.class_name, json.dumps(case_id))
        }
        try:
            res = self._request("POST", "/v1/graphql", gql)
            rows = (
                res.get("data", {})
                .get("Aggregate", {})
                .get(self.class_name, [])
            )
            if rows and isinstance(rows, list):
                meta = rows[0].get("meta") or {}
                return int(meta.get("count") or 0)
            return 0
        except Exception:
            return 0

    def list_evidence_ids(self, case_id: str, *, page_size: int = 500) -> set[str]:
        out: set[str] = set()
        offset = 0
        while True:
            gql = {
                "query": """
                {
                  Get {
                    %s(
                      where: {path: [\"case_id\"], operator: Equal, valueText: %s}
                      limit: %d
                      offset: %d
                    ) {
                      evidence_id
                    }
                  }
                }
                """
                % (
                    self.class_name,
                    json.dumps(case_id),
                    int(page_size),
                    int(offset),
                )
            }
            try:
                res = self._request("POST", "/v1/graphql", gql)
            except Exception:
                break

            rows = (
                res.get("data", {})
                .get("Get", {})
                .get(self.class_name, [])
            )
            if not rows:
                break

            for r in rows:
                evid = r.get("evidence_id")
                if evid:
                    out.add(str(evid))

            if len(rows) < page_size:
                break
            offset += page_size

        return out

    def delete_all(self, case_id: str) -> None:
        """Deletes all objects for a given case_id using Weaviate's batch delete API."""
        payload = {
            "match": {
                "class": self.class_name,
                "where": {
                    "path": ["case_id"],
                    "operator": "Equal",
                    "valueText": case_id
                }
            }
        }
        try:
            res = self._request("DELETE", "/v1/batch/objects", payload)
            results = res.get("results", {})
            successful = results.get("successful", 0)
            failed = results.get("failed", 0)
            print(f"[INFO] Deleted {successful} items for case_id '{case_id}' in {self.class_name}. (Failed: {failed})")
        except Exception as e:
            print(f"[ERROR] Failed to delete items for case_id '{case_id}': {e}")

    def ensure_schema(self) -> None:
        schema = self._request("GET", "/v1/schema")
        classes = {c.get("class") for c in schema.get("classes", [])}
        if self.class_name in classes:
            return

        payload = {
            "class": self.class_name,
            "description": "Evidence chunks for legal-agent mock trials",
            "vectorizer": (self.vectorizer or "none"),
            "properties": [
                {"name": "case_id", "dataType": ["text"]},
                {"name": "evidence_id", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
            ],
        }
        try:
            self._request("POST", "/v1/schema", payload)
        except Exception:
            # Common failure: configured vectorizer module isn't enabled on the Weaviate instance.
            if payload.get("vectorizer") != "none":
                payload["vectorizer"] = "none"
                self._request("POST", "/v1/schema", payload)
            else:
                raise

    def _uuid_for(self, case_id: str, evidence_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{case_id}:{evidence_id}"))

    def upsert_all(self, case_id: str, items: Iterable[EvidenceItem], batch_size: int = 40) -> None:
        self.ensure_schema()
        
        # Convert to list to handle batching
        items_list = list(items)
        if not items_list:
            return

        total = len(items_list)
        print(f"[INFO] Starting ingestion of {total} items into {self.class_name}...")

        for i in range(0, len(items_list), batch_size):
            batch = items_list[i : i + batch_size]
            objects = []
            for it in batch:
                obj_id = self._uuid_for(case_id, it.evidence_id)
                objects.append({
                    "class": self.class_name,
                    "id": obj_id,
                    "properties": {
                        "case_id": case_id,
                        "evidence_id": it.evidence_id,
                        "text": it.text,
                    },
                })
            
            # Weaviate Batch API: POST /v1/batch/objects
            try:
                self._request("POST", "/v1/batch/objects", {"objects": objects})
                print(f"  [INFO] Ingested {min(i + batch_size, total)}/{total} items...")
            except Exception as e:
                # Fallback to individual upserts if batching fails for some reason
                print(f"  [WARN] Batch {i//batch_size + 1} failed ({e}), falling back to individual requests...")
                for j, it in enumerate(batch):
                    obj_id = self._uuid_for(case_id, it.evidence_id)
                    payload = {
                        "class": self.class_name,
                        "id": obj_id,
                        "properties": {
                            "case_id": case_id,
                            "evidence_id": it.evidence_id,
                            "text": it.text,
                        },
                    }
                    try:
                        self._request("POST", "/v1/objects", payload)
                        if (j + 1) % 10 == 0 or (i + j + 1) == total:
                            print(f"    [INFO] Progress: {i + j + 1}/{total} items...")
                    except RuntimeError as re:
                        if "already" in str(re).lower() and "exist" in str(re).lower():
                            self._request("PUT", f"/v1/objects/{obj_id}", payload)
                        else:
                            raise re

    def get(self, case_id: str, evidence_id: str) -> Optional[EvidenceItem]:
        _ = case_id
        obj_id = self._uuid_for(case_id, evidence_id)
        try:
            obj = self._request("GET", f"/v1/objects/{obj_id}")
            props = obj.get("properties") or {}
            text = props.get("text")
            if not text:
                return None
            return EvidenceItem(evidence_id=evidence_id, text=text)
        except (HTTPError, URLError, TimeoutError, RuntimeError):
            return None

    def search(self, case_id: str, query: str, k: int = 5) -> List[EvidenceItem]:
        def _hits_from(gql: dict) -> List[EvidenceItem]:
            res = self._request("POST", "/v1/graphql", gql)
            if "errors" in res and res["errors"]:
                raise RuntimeError(f"Weaviate GraphQL error: {res['errors']}")
            hits = (
                res.get("data", {})
                .get("Get", {})
                .get(self.class_name, [])
            )
            if hits is None:
                return []
            out: List[EvidenceItem] = []
            for h in hits:
                evid = h.get("evidence_id")
                text = h.get("text")
                if evid and text:
                    out.append(EvidenceItem(evidence_id=evid, text=text))
            return out

        # Try semantic search first; fall back to BM25 for non-vectorized Weaviate.
        gql_near = {
            "query": """
            {
              Get {
                %s(
                  where: {path: [\"case_id\"], operator: Equal, valueText: %s}
                  nearText: {concepts: [%s]}
                  limit: %d
                ) {
                  evidence_id
                  text
                }
              }
            }
            """
            % (
                self.class_name,
                json.dumps(case_id),
                json.dumps(query),
                int(k),
            )
        }

        try:
            return _hits_from(gql_near)
        except Exception:
            gql_bm25 = {
                "query": """
                {
                  Get {
                    %s(
                      where: {path: [\"case_id\"], operator: Equal, valueText: %s}
                      bm25: {query: %s}
                      limit: %d
                    ) {
                      evidence_id
                      text
                    }
                  }
                }
                """
                % (
                    self.class_name,
                    json.dumps(case_id),
                    json.dumps(query),
                    int(k),
                )
            }
            try:
                return _hits_from(gql_bm25)
            except Exception:
                return []
