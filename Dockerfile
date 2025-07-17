FROM python:3.10-slim

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the parser and dashboard code; mount bdd100k_labels at runtime
EXPOSE 8050

CMD ["python", "src/dashboard.py"]
