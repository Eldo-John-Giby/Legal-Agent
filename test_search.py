import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
key = os.getenv('WEAVIATE_API_KEY')

gql = {
    'query': '''
    {
      Get {
        SC_Precedents(bm25: {query: "fundamental rights"}, limit: 3) {
          case_name
          year
          chunk_id
          source_pdf
        }
      }
    }
    '''
}

req = urllib.request.Request(
    f'{url}/v1/graphql',
    data=json.dumps(gql).encode('utf-8'),
    method='POST'
)
req.add_header('Authorization', f'Bearer {key}')
req.add_header('Content-Type', 'application/json')

try:
    with urllib.request.urlopen(req) as resp:
        print(json.dumps(json.loads(resp.read()), indent=2))
except Exception as e:
    print(f"Error: {e}")
