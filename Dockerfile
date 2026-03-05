# 1. Use a lightweight Python base image
FROM python:3.14.3-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependencies file first (for caching)
COPY requirements.txt .

# 4. Install dependencies (CPU version of PyTorch to save space)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code and model
COPY model.py .
COPY app.py .
# Install curl, then download the weights directly from your GitHub Release
# Replace the old COPY line with these:
RUN apt-get update && apt-get install -y curl
RUN curl -L -o best_densenet121.pth "https://github.com/DRAGOX7/chest-xray-mlops/releases/download/v1.0.0/best_densenet121.pth"

# 6. Expose the port that FastAPI uses
EXPOSE 8000

# 7. The command to run when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
