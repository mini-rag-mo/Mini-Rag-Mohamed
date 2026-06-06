cat > docker/minirag/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/models/db_schemes/minirag/

python3 -c "
import re, os
url = os.environ.get('POSTGRES_URL', '')
with open('alembic.ini', 'r') as f:
    content = f.read()
content = re.sub(r'sqlalchemy\.url = .*', 'sqlalchemy.url = ' + url, content)
with open('alembic.ini', 'w') as f:
    f.write(content)
"

alembic upgrade head
cd /app

exec "\$@"
EOF