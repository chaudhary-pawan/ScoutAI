# 1. Lighter version of Python
FROM python:3.9-slim

# 2. folder inside container
WORKDIR /app

# 3. copy requirements 
COPY requirements.txt .

# 4. install Libraries without cache
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy whole codebase
COPY . .

# 6. Tell render about the port we're using
EXPOSE 10000

# 7. Server Start Command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]