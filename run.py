#!/usr/bin/env python
from website import app

# Build logger if doesn't exist
import os

if not os.path.exists('/tmp/log.csv'):
    os.mknod('/tmp/log.csv')

app.run(debug=True)
