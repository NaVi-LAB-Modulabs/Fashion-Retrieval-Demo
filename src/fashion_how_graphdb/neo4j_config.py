"""Neo4j connection defaults for read-only retrieval."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
