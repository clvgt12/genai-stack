FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# RUN pip install --upgrade -r requirements.txt
RUN pip install -r requirements.txt

COPY ask_your_docx.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8513

HEALTHCHECK CMD curl --fail http://localhost:8513/_stcore/health

ENTRYPOINT ["streamlit", "run", "ask_your_docx.py", "--server.port=8513", "--server.address=0.0.0.0"]
