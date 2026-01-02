"""
Repositories Layer
Data persistence and query operations for GP Data v4.
"""
from .connection import db_manager, get_database, DatabaseManager
from .leads import LeadRepository
from .messages import MessageRepository
from .base import BaseRepository

__all__ = [
    "db_manager",
    "get_database",
    "DatabaseManager",
    "LeadRepository",
    "MessageRepository",
    "BaseRepository",
]
