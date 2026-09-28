"""
Vercel serverless entrypoint for the lyrics -> genre classifier website.

Vercel's Python runtime looks for a WSGI/ASGI app object in api/*.py and
runs each file as its own serverless function. This file re-exports the
same Flask app defined in SITE/app.py so the deployed site and the
locally-run site (`python SITE/app.py`) share one implementation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SITE"))

from app import app  # noqa: E402  (SITE/app.py's Flask instance)
