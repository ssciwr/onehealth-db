# ----------------------------------------------
# ------      POSTGRESQL RDBM SANDBOX      -----
# ----------------------------------------------
# Authors:
#   OneHealth Platform Team
#	Scientific Software Center
# Last update: 30.04.2025


# Use the official PostgreSQL image from Docker Hub
FROM postgres:latest

# Set environment variables for PostgreSQL
#ENV POSTGRES_USER=postgres
#ENV POSTGRES_PASSWORD=postgres
#ENV POSTGRES_DB=postgres

# Copy the SQL file from your host machine into the container
# This assumes your SQL file is located at the root of your project
#COPY ./sql/initial_db_setup.sql /docker-entrypoint-initdb.d/
#COPY ./_scripts/sql/initial_db_sandbox_setup.sql /docker-entrypoint-initdb.d/init.sql

# Expose the default PostgreSQL port
EXPOSE 5432
