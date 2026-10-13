# Lean image for free hosting (no torch). Works on Render and Hugging Face Spaces.
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements-render.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements-render.txt
COPY backend/ ./backend/
COPY --from=web /web/dist ./frontend/dist
RUN chmod -R 777 /app/backend
WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]