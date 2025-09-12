# ----------------------------------------------
# ------      Backend build    -----
# ----------------------------------------------
# Authors:
#   OneHealth Platform Team
#	Scientific Software Center
# Last update: 28.07.2025
FROM python:3.11-slim AS application

# Install system dependencies for PostgreSQL client and geospatial libraries
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files and install Python dependencies
COPY onehealth_db ./onehealth_db
COPY README.md .
COPY pyproject.toml .
RUN pip install --no-cache-dir .


# Create entrypoint script for flexibility
RUN cat > /onehealth_db/entrypoint.sh <<'EOF'
#!/bin/bash
# Execute the requested command
case "$1" in

    "production")
        exec python -m onehealth_db.production
        ;;
    "api")
        exec uvicorn onehealth_db.main:app --host 0.0.0.0 --port 8000
        ;;
    *)
        exec "$@"
        ;;
esac
EOF

# Make entrypoint executable and create non-root user
RUN chmod +x /onehealth_db/entrypoint.sh 

# Set environment variables
ENV PYTHONPATH=/onehealth_db
ENV CONFIG_FILE=/onehealth_db/production.yaml

# Expose port for FastAPI
EXPOSE 8000

ENTRYPOINT ["/onehealth_db/entrypoint.sh"]
CMD ["api"]