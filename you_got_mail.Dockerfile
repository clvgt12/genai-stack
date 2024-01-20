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

COPY you_got_mail.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8510

HEALTHCHECK CMD curl --fail http://localhost:8510/_stcore/health

ENTRYPOINT ["streamlit", "run", "you_got_mail.py", "--server.port=8510", "--server.address=0.0.0.0"]
