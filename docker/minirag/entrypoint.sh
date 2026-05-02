#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/models/db_schemes/minirag/

# Replace the database URL in alembic.ini with the environment variable
sed -i "s|sqlalchemy.url = .*|sqlalchemy.url = ${POSTGRES_URL}|" alembic.ini

alembic upgrade head
cd /app

exec "$@"
