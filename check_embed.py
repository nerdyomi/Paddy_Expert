
from src.ingestion.embedder import embed
vecs = embed(['rice blast disease control'], input_type='query')
print(len(vecs), len(vecs[0]))
