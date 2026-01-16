import time
from main import rag_pipeline
start = time.time()
rag_pipeline("Tell me about trek clean up drives")
print("Latency:", time.time() - start)
