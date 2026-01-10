FROM python:3.11-slim

WORKDIR /app

# Ferramentas de rede + debug
RUN apt-get update && apt-get install -y \
    iputils-ping \
    curl \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "run_node.py", "5000", "--poison-rate", "1", "--delay", "30"]
