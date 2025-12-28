# Base image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

# Run GUI using the package module entrypoint (the repository does not include `scripts/gui_app.py`)
CMD ["python", "-m", "src.gui.app"]