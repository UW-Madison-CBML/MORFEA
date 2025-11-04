FROM python:3.9-slim
WORKDIR /app
COPY train_requirements.txt .
COPY . .  # copies from build context into /app
RUN pip install --no-cache-dir -r train_requirements.txt
CMD ["python", "your_app_file.py"]
