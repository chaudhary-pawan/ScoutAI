import time
from main import rag_pipeline

start = time.time()

answer = rag_pipeline("Tell me about trek clean up drives")
print("\nANSWER:\n", answer)

print("\nLatency:", round(time.time() - start, 2), "seconds")
