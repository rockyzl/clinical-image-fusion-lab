FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.163.2 uvicorn==0.54.0 numpy==2.5.5 pillow==12.3.0 onnxruntime==1.32.0
COPY app /app/app
RUN useradd --create-home appuser && mkdir /app/.runtime && chown appuser /app/.runtime
USER appuser
EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
