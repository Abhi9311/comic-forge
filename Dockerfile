# Stage 1: build the React frontend
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend + local image model (CPU)
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ ./backend/
COPY --from=web /web/dist ./frontend/dist
RUN mkdir -p /app/backend/storage && chown -R user:user /app
USER user
ENV HF_HOME=/home/user/hf
RUN python -c "from diffusers import AutoPipelineForText2Image as A; A.from_pretrained('stabilityai/sd-turbo', variant='fp16')"
WORKDIR /app/backend
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]