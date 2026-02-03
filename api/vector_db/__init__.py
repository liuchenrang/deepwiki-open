"""
向量数据库抽象层

支持多种向量数据库后端：
- FAISS (本地文件存储)
- pgvector (PostgreSQL + pgvector 扩展)
"""

from .base import VectorDBBackend, SearchResult, DistanceMetric
from .factory import get_vector_db, VectorDBConfig
from .faiss_backend import FaissBackend

__all__ = [
    "VectorDBBackend",
    "SearchResult",
    "DistanceMetric",
    "get_vector_db",
    "VectorDBConfig",
    "FaissBackend",
]

# 动态导入 pgvector（如果安装）
try:
    from .pgvector_backend import PgvectorBackend
    __all__.append("PgvectorBackend")
except ImportError:
    pass
