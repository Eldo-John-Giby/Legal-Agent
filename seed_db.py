import argparse
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from courtroom_ai
sys.path.append(str(Path(__file__).parent.resolve()))

from courtroom_ai.constitution import parse_constitution_text, parse_constitution_text_fast
from courtroom_ai.rag import WeaviateEvidenceStore, parse_case_text, parse_precedents_text
from courtroom_ai.runner import _build_model_client


async def seed_database(
    constitution_path: Path,
    case_path: Path,
    precedents_path: Path,
    *,
    force_constitution: bool = False,
    constitution_mode: str = "fast",
    constitution_start_segment: int = 1,
    constitution_stop_on_error: bool = False,
) -> None:
    load_dotenv()

    print("=== Courtroom AI Database Seeder ===")
    
    # 1. Verify Weaviate Availability
    weaviate_store = WeaviateEvidenceStore(class_name="EvidenceChunk")
    if not weaviate_store.is_available():
        print(f"[ERROR] Weaviate is not reachable at: {weaviate_store.base_url}")
        print("Please ensure Weaviate is running and WEAVIATE_URL is set correctly in your .env file.")
        sys.exit(1)
    
    print(f"[INFO] Connected to Weaviate at: {weaviate_store.base_url}")

    # Fail fast (before any expensive LLM calls) by validating schemas exist/are creatable.
    try:
        weaviate_store.ensure_schema()
        WeaviateEvidenceStore(class_name="ConstitutionChunk").ensure_schema()
        WeaviateEvidenceStore(class_name="AuthorityChunk").ensure_schema()
    except Exception as e:
        print(f"[ERROR] Weaviate schema setup failed: {e}")
        print("Fix WEAVIATE_URL/WEAVIATE_VECTORIZER (or enable a vectorizer module), then retry.")
        sys.exit(1)

    # 2. Parse & Embed Constitution
    if constitution_path.exists():
        const_store = WeaviateEvidenceStore(class_name="ConstitutionChunk")

        constitution_text = constitution_path.read_text(encoding="utf-8")

        if (constitution_mode or "fast").lower() == "llm":
            existing = const_store.count("shared_constitution")
            if existing > 0 and not force_constitution and constitution_start_segment <= 1:
                print(f"[INFO] ConstitutionChunk already has {existing} items for shared_constitution; skipping re-embed.")
            else:
                print(f"[INFO] Parsing constitution from: {constitution_path}")

                print("[INFO] Initializing LLM client for parsing...")
                client = _build_model_client()
                print("[INFO] Asking LLM to parse and chunk the constitution...")
                print("[INFO] Streaming ingest into 'ConstitutionChunk' during parsing...")

                ingested = 0
                existing_ids = const_store.list_evidence_ids("shared_constitution") if not force_constitution else set()

                async def _ingest_chunk(chunk_items):
                    nonlocal ingested
                    chunk_items = [it for it in chunk_items if it.evidence_id not in existing_ids]
                    if not chunk_items:
                        return
                    const_store.upsert_all(case_id="shared_constitution", items=chunk_items)
                    for it in chunk_items:
                        existing_ids.add(it.evidence_id)
                    ingested += len(chunk_items)
                    print(f"[INFO] Ingested {ingested} constitution articles so far...")

                await parse_constitution_text(
                    constitution_text,
                    client,
                    on_items=_ingest_chunk,
                    start_chunk=max(0, int(constitution_start_segment) - 1),
                    continue_on_error=not constitution_stop_on_error,
                )
                print("[SUCCESS] Constitution ingest run completed.")

        else:
            # Resume-friendly path: parse locally and only ingest missing ids.
            fast_items = parse_constitution_text_fast(constitution_text)
            if fast_items:
                existing_ids = const_store.list_evidence_ids("shared_constitution") if not force_constitution else set()
                missing = [it for it in fast_items if it.evidence_id not in existing_ids]
                if existing_ids and not missing:
                    print(f"[INFO] Constitution already fully ingested ({len(existing_ids)} items); skipping.")
                else:
                    if existing_ids:
                        print(f"[INFO] Resuming constitution ingest: {len(existing_ids)} present, {len(missing)} missing.")
                    else:
                        print(f"[INFO] Ingesting constitution (fast parser): {len(fast_items)} articles.")
                    const_store.upsert_all(case_id="shared_constitution", items=missing if existing_ids else fast_items)
                    print("[SUCCESS] Constitution articles ingested.")
            else:
                raise SystemExit(
                    "Constitution fast parser found 0 articles; rerun with --constitution-mode llm."
                )
    else:
        print(f"[WARN] Constitution file not found at {constitution_path}. Skipping.")

    # 3. Parse & Embed Legal Precedents
    if precedents_path.exists():
        auth_store = WeaviateEvidenceStore(class_name="AuthorityChunk")
        precedents_text = precedents_path.read_text(encoding="utf-8")
        items = parse_precedents_text(precedents_text)
        if items:
            print(f"[INFO] Ingesting {len(items)} legal precedents...")
            auth_store.upsert_all(case_id="shared_precedents", items=items)
            print("[SUCCESS] Legal precedents ingested.")
    else:
        print(f"[WARN] Precedents file not found at {precedents_path}. Skipping.")

    # 4. Parse & Embed Case Evidence
    if case_path.exists():
        print(f"[INFO] Parsing case data from: {case_path}")
        case_text = case_path.read_text(encoding="utf-8")
        summary, evidence_items = parse_case_text(case_text)
        case_id = str(case_path.resolve())

        if evidence_items:
            print(f"[INFO] Found {len(evidence_items)} evidence items for case: '{summary.title or case_path.name}'.")
            print("Ingesting into 'EvidenceChunk'...")
            weaviate_store.upsert_all(case_id=case_id, items=evidence_items)
            print("[SUCCESS] Case evidence uploaded and embedded successfully!")
        else:
            print("[WARN] Case file parsed to 0 evidence items.")
    else:
        print(f"[WARN] Case file not found at {case_path}. Skipping.")

    print("\n=== Seeding Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Weaviate with constitution and case evidence chunks.")
    parser.add_argument(
        "--constitution",
        default="data/constitution.txt",
        help="Path to the constitution text file."
    )
    parser.add_argument(
        "--case",
        default="data/sample_case.txt",
        help="Path to the case text file."
    )
    parser.add_argument(
        "--precedents",
        default="data/precedents.txt",
        help="Path to the precedents text file."
    )
    parser.add_argument(
        "--force-constitution",
        action="store_true",
        help="Re-parse and re-embed the constitution even if it's already in Weaviate."
    )
    parser.add_argument(
        "--constitution-mode",
        choices=["fast", "llm"],
        default="fast",
        help="How to parse constitution: 'fast' (local) or 'llm' (slow/costly, chunked by model)."
    )
    parser.add_argument(
        "--constitution-start-segment",
        type=int,
        default=1,
        help="(LLM mode) 1-based segment number to start from (e.g. 199 to resume)."
    )
    parser.add_argument(
        "--constitution-stop-on-error",
        action="store_true",
        help="(LLM mode) Stop immediately when a segment fails after retries."
    )
    args = parser.parse_args()

    asyncio.run(
        seed_database(
            Path(args.constitution),
            Path(args.case),
            Path(args.precedents),
            force_constitution=args.force_constitution,
            constitution_mode=args.constitution_mode,
            constitution_start_segment=args.constitution_start_segment,
            constitution_stop_on_error=args.constitution_stop_on_error,
        )
    )
