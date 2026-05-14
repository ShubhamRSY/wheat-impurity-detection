FROM pytorch/pytorch:2.1.0-cpu

WORKDIR /app

RUN pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy Pillow prometheus-client

COPY models/wheat_segmentation_v2.pt models/
COPY deployment/api_server.py .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
