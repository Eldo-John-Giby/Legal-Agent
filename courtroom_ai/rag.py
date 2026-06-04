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
    def upsert_all(self, case_id: str, items: Iterable[EvidenceItem]) -> None:
        raise NotImplementedError

    def search(self, case_id: str, query: str, k: int = 5) -> List[EvidenceItem]:
        raise NotImplementedError

    def get(self, case_id: str, evidence_id: str) -> Optional[EvidenceItem]:
        raise NotImplementedError


class LocalEvidenceStore(EvidenceStore):
    def __init__(self, items: List[EvidenceItem]):
        self._items = items
        self._by_id = {e.evidence_id: e for e in items}

    def upsert_all(self, case_id: str, items: Iterable[EvidenceItem]) -> None:
        _ = case_id, items

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
        self.vectorizer = vectorizer or os.getenv("WEAVIATE_VECTORIZER", "text2vec-transformers")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            url=f"{self.base_url}{path}",
            method=method,
            data=data,
            headers=self._headers(),
        )
        with urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}

    def is_available(self) -> bool:
        try:
            self._request("GET", "/v1/meta")
            return True
        except Exception:
            return False

    def ensure_schema(self) -> None:
        schema = self._request("GET", "/v1/schema")
        classes = {c.get("class") for c in schema.get("classes", [])}
        if self.class_name in classes:
            return

        payload = {
            "class": self.class_name,
            "description": "Evidence chunks for legal-agent mock trials",
            "vectorizer": self.vectorizer,
            "properties": [
                {"name": "case_id", "dataType": ["text"]},
                {"name": "evidence_id", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
            ],
        }
        self._request("POST", "/v1/schema", payload)

    def _uuid_for(self, case_id: str, evidence_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{case_id}:{evidence_id}"))

    def upsert_all(self, case_id: str, items: Iterable[EvidenceItem]) -> None:
        self.ensure_schema()
        for it in items:
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
            # PUT is idempotent upsert.
            self._request("PUT", f"/v1/objects/{self.class_name}/{obj_id}", payload)

    def get(self, case_id: str, evidence_id: str) -> Optional[EvidenceItem]:
        _ = case_id
        obj_id = self._uuid_for(case_id, evidence_id)
        try:
            obj = self._request("GET", f"/v1/objects/{self.class_name}/{obj_id}")
            props = obj.get("properties") or {}
            text = props.get("text")
            if not text:
                return None
            return EvidenceItem(evidence_id=evidence_id, text=text)
        except (HTTPError, URLError, TimeoutError):
            return None

    def search(self, case_id: str, query: str, k: int = 5) -> List[EvidenceItem]:
        gql = {
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
            res = self._request("POST", "/v1/graphql", gql)
            hits = (
                res.get("data", {})
                .get("Get", {})
                .get(self.class_name, [])
            )
            out: List[EvidenceItem] = []
            for h in hits:
                evid = h.get("evidence_id")
                text = h.get("text")
                if evid and text:
                    out.append(EvidenceItem(evidence_id=evid, text=text))
            return out
        except Exception:
            return []
