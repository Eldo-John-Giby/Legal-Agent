from courtroom_ai.rag import WeaviateEvidenceStore
import os
from dotenv import load_dotenv

def verify_precedents():
    load_dotenv()
    # Assuming the class name for precedents is 'PrecedentChunk' based on seed_db.py logic
    store = WeaviateEvidenceStore(class_name="PrecedentChunk")
    
    # Check total count
    # Note: precedents might not use a specific 'case_id' like 'shared_constitution' 
    # but seed_db.py uses 'shared_precedents'
    count = store.count("shared_precedents")
    print(f"Current count in Weaviate for 'shared_precedents': {count}")
    
    # Fetch a sample to see metadata structure
    results = store.search("shared_precedents", "contract breach", k=1)
    if results:
        print("\nSample Precedent:")
        print(f"ID: {results[0].evidence_id}")
        print(f"Text Snippet: {results[0].text[:200]}...")
    else:
        print("\nNo precedents found in search.")

if __name__ == "__main__":
    verify_precedents()
