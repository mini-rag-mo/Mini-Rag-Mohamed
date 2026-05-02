#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/models/db_schemes/minirag/

# Replace the database URL in alembic.ini with the environment variable
python3 -c "
import re, os
with open('alembic.ini', 'r') as f:
    content = f.read()
content = re.sub(r'sqlalchemy\.url = .*', 'sqlalchemy.url = ' + os.environ['POSTGRES_URL'], content)
with open('alembic.ini', 'w') as f:
    f.write(content)
"
alembic upgrade head
cd /app

exec "$@"
