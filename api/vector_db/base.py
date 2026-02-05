"""
向量数据库抽象基类

定义统一的向量数据库接口，支持多种后端实现。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from adalflow.core.types import Document


class DistanceMetric(Enum):
    """距离度量类型"""
    L2 = "l2"           # 欧几里得距离
    INNER_PRODUCT = "inner_product"  # 内积
    COSINE = "cosine"   # 余弦相似度


@dataclass
class SearchResult:
    """搜索结果"""
    document: Document          # 匹配的文档
    score: float                # 相似度分数
    distance: Optional[float]   # 距离（可选）


class VectorDBBackend(ABC):
    """
    向量数据库后端抽象接口

    所有向量数据库后端必须实现此接口。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量数据库后端

        Args:
            config: 配置字典
        """
        self.config = config
        self.embedding_dimension = config.get("embedding_dimension", 1024)

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        添加文档及其向量到数据库

        Args:
            documents: 文档列表（必须包含 vector 字段）
            **kwargs: 额外参数

        Returns:
            文档 ID 列表
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> int:
        """
        删除指定 ID 的文档

        Args:
            document_ids: 文档 ID 列表

        Returns:
            删除的文档数量
        """
        pass

    @abstractmethod
    def delete_by_metadata(self, key: str, value: Any, soft_delete: bool = True) -> int:
        """
        根据元数据删除文档

        Args:
            key: 元数据键
            value: 元数据值
            soft_delete: 是否使用软删除（默认 True）
                        - True: 标记为已删除（deleted = TRUE）
                        - False: 物理删除（从数据库中移除记录）
                        注意：FAISS 仅支持软删除

        Returns:
            删除的文档数量
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> List[SearchResult]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量
            top_k: 返回前 K 个结果
            filters: 元数据过滤条件（可选）
            distance_metric: 距离度量类型

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        获取单个文档

        Args:
            document_id: 文档 ID

        Returns:
            文档对象，如果不存在返回 None
        """
        pass

    @abstractmethod
    def get_documents_by_metadata(self, metadata: Dict[str, Any]) -> List[Document]:
        """
        根据元数据获取文档

        Args:
            metadata: 元数据字典

        Returns:
            文档列表
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document: Document) -> bool:
        """
        更新文档

        Args:
            document_id: 文档 ID
            document: 新的文档对象

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        统计文档数量

        Args:
            filters: 可选的过滤条件

        Returns:
            文档数量
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存数据库到磁盘

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从磁盘加载数据库

        Args:
            path: 加载路径
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭数据库连接，释放资源"""
        pass

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
