"""
向量数据库工厂方法

根据配置创建向量数据库实例。
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from api.vector_db.base import VectorDBBackend


logger = logging.getLogger(__name__)


@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    backend: str  # "faiss" 或 "pgvector"
    config: Dict[str, Any]

    @classmethod
    def from_env(cls) -> "VectorDBConfig":
        """从环境变量加载配置"""
        backend = os.getenv("VECTOR_DB_BACKEND", "faiss")

        config = {}

        if backend == "pgvector":
            config = {
                "connection_string": os.getenv(
                    "DATABASE_URL",
                    "postgresql://deepwiki:deepwiki@localhost:5432/deepwiki"
                ),
                "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "1024")),
                "index_type": os.getenv("PGVECTOR_INDEX_TYPE", "hnsw"),
                "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            }
        elif backend == "faiss":
            config = {
                "index_path": os.getenv(
                    "FAISS_INDEX_PATH",
                    os.path.expanduser("~/.adalflow/databases/")
                ),
                "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "1024")),
            }
        else:
            raise ValueError(f"Unsupported vector database backend: {backend}")

        return cls(backend=backend, config=config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorDBConfig":
        """从字典加载配置"""
        backend = config_dict.get("backend", "faiss")

        if backend == "pgvector":
            # pgvector 配置
            pgvector_config = config_dict.get("pgvector", {})
            config = {
                "connection_string": pgvector_config.get(
                    "connection_string",
                    "postgresql://deepwiki:deepwiki@localhost:5432/deepwiki"
                ),
                "embedding_dimension": pgvector_config.get("embedding_dimension", 1024),
                "index_type": pgvector_config.get("index_type", "hnsw"),
                "index_parameters": pgvector_config.get("index_parameters", {
                    "m": 16,
                    "ef_construction": 64
                }),
                "pool_size": pgvector_config.get("pool_size", 10),
            }
        elif backend == "faiss":
            # FAISS 配置
            faiss_config = config_dict.get("faiss", {})
            config = {
                "index_path": faiss_config.get(
                    "index_path",
                    os.path.expanduser("~/.adalflow/databases/")
                ),
                "embedding_dimension": faiss_config.get("embedding_dimension", 1024),
                "index_type": faiss_config.get("index_type", "Flat"),
            }
        else:
            raise ValueError(f"Unsupported vector database backend: {backend}")

        return cls(backend=backend, config=config)


def get_vector_db(config: Any = None) -> VectorDBBackend:
    """
    创建向量数据库实例

    Args:
        config: 配置对象，可以是：
            - VectorDBConfig 实例
            - 配置字典
            - None（从环境变量加载）

    Returns:
        向量数据库实例

    Raises:
        ValueError: 不支持的后端类型
        ImportError: 缺少必要的依赖
    """
    # 规范化配置
    if config is None:
        vector_db_config = VectorDBConfig.from_env()
    elif isinstance(config, dict):
        vector_db_config = VectorDBConfig.from_dict(config)
    elif isinstance(config, VectorDBConfig):
        vector_db_config = config
    else:
        raise TypeError(f"Invalid config type: {type(config)}")

    backend = vector_db_config.backend
    logger.info(f"Initializing vector database backend: {backend}")

    # 创建对应的后端实例
    if backend == "faiss":
        from api.vector_db.faiss_backend import FaissBackend
        # 添加 backend 标识到 config
        faiss_config = vector_db_config.config.copy()
        faiss_config['backend'] = 'faiss'
        return FaissBackend(faiss_config)

    elif backend == "pgvector":
        try:
            from api.vector_db.pgvector_backend import PgvectorBackend
            # 添加 backend 标识到 config
            pgvector_config = vector_db_config.config.copy()
            pgvector_config['backend'] = 'pgvector'
            return PgvectorBackend(pgvector_config)
        except ImportError as e:
            logger.error(f"Failed to import pgvector backend: {e}")
            logger.error("Please install required dependencies:")
            logger.error("  pip install psycopg[binary] pgvector")
            raise ImportError(
                "pgvector backend requires psycopg and pgvector packages. "
                "Install them with: pip install psycopg[binary] pgvector"
            )

    else:
        raise ValueError(
            f"Unsupported vector database backend: {backend}. "
            f"Supported backends: faiss, pgvector"
        )
