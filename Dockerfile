FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python make_sample_data.py && useradd -m app && chown -R app /app
USER app
ENV PORT=8000 PROOFAI_DATA=/tmp/proofai
EXPOSE 8000
CMD ["python", "server.py"]
