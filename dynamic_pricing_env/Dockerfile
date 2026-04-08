FROM openenv-base:latest

WORKDIR /app

# Copy root files first
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY spaces.yaml /app/spaces.yaml

# Copy environment source
COPY dynamic_pricing_env /app/dynamic_pricing_env

# Install dependencies (includes gradio, fastapi, uvicorn, etc.)
RUN pip install -r requirements.txt

# Expose ports
EXPOSE 7860 8000

# Run startup script that starts both FastAPI (background) and Gradio (foreground)
RUN echo '#!/bin/bash\n\
cd /app/dynamic_pricing_env\n\
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &\n\
sleep 2\n\
cd /app\n\
python app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
