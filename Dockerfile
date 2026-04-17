FROM python:3.12-slim


WORKDIR /app

# Install deps first — this layer is cached across source-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Editable install so BASEDIR resolves to /app (not site-packages).
COPY  . .

RUN pip install --no-cache-dir --no-deps -e .

# Create writable dirs for runtime artifacts and seed their permissions.
# mkdir must run before chmod so these dirs exist at the right permission level;
# named Docker volumes are initialised from the image content, so /app/.cache
# must already be 777 here or the volume mount will be root-owned.
RUN mkdir -p /app/.cache /app/models /app/output && chmod -R 777 /app


ENTRYPOINT ["u2demo-forecast"]
