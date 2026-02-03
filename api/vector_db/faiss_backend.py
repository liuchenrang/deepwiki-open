"""
FAISS 向量数据库后端

封装 AdalFlow 的 LocalDB，实现向量数据库抽象接口。
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from adalflow.core.db import LocalDB
from adalflow.core.types import Document

from api.vector_db.base import (
    VectorDBBackend,
    SearchResult,
    DistanceMetric
)


logger = logging.getLogger(__name__)


class FaissBackend(VectorDBBackend):
    """
    FAISS 向量数据库后端

    使用 AdalFlow 的 LocalDB 实现，保持与现有代码的兼容性。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 FAISS 后端

        Args:
            config: 配置字典
                - index_path: 索引文件存储路径
                - embedding_dimension: 向量维度
                - index_type: FAISS 索引类型（Flat, IVF, HNSW 等）
        """
        super().__init__(config)
        self.index_path = config.get("index_path", os.path.expanduser("~/.adalflow/databases/"))
        self.index_type = config.get("index_type", "Flat")

        # 确保目录存在
        os.makedirs(self.index_path, exist_ok=True)

        # AdalFlow LocalDB 实例
        self.db: Optional[LocalDB] = None
        self.documents: List[Document] = []

        # 文档 ID 映射（用于删除和更新）
        self.doc_id_map: Dict[str, int] = {}  # document_id -> index
        self.next_doc_id = 0

        logger.info(f"FAISS backend initialized with index_path={self.index_path}")

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        添加文档到数据库

        Args:
            documents: 文档列表（必须包含 vector 字段）

        Returns:
            文档 ID 列表
        """
        if not self.db:
            self.db = LocalDB()

        document_ids = []

        for doc in documents:
            # 生成文档 ID
            doc_id = f"doc_{self.next_doc_id}"
            self.next_doc_id += 1

            # 添加元数据
            if not hasattr(doc, 'meta_data') or doc.meta_data is None:
                doc.meta_data = {}
            doc.meta_data['_doc_id'] = doc_id

            # 存储文档
            self.documents.append(doc)
            self.doc_id_map[doc_id] = len(self.documents) - 1
            document_ids.append(doc_id)

        logger.info(f"Added {len(documents)} documents to FAISS backend")
        return document_ids

    def delete_documents(self, document_ids: List[str]) -> int:
        """
        删除文档（软删除，标记为已删除）

        注意：FAISS 不支持真正的删除，这里使用软删除标记。

        Args:
            document_ids: 文档 ID 列表

        Returns:
            删除的文档数量
        """
        deleted_count = 0

        for doc_id in document_ids:
            if doc_id in self.doc_id_map:
                # 标记为已删除（在元数据中）
                index = self.doc_id_map[doc_id]
                if index < len(self.documents):
                    if not hasattr(self.documents[index], 'meta_data'):
                        self.documents[index].meta_data = {}
                    self.documents[index].meta_data['_deleted'] = True
                    deleted_count += 1
                del self.doc_id_map[doc_id]

        logger.info(f"Soft deleted {deleted_count} documents")
        return deleted_count

    def delete_by_metadata(self, key: str, value: Any) -> int:
        """
        根据元数据删除文档

        Args:
            key: 元数据键
            value: 元数据值

        Returns:
            删除的文档数量
        """
        doc_ids_to_delete = []

        for doc_id, index in self.doc_id_map.items():
            if index >= len(self.documents):
                continue

            doc = self.documents[index]
            if hasattr(doc, 'meta_data') and doc.meta_data:
                if doc.meta_data.get(key) == value:
                    doc_ids_to_delete.append(doc_id)

        return self.delete_documents(doc_ids_to_delete)

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
            filters: 元数据过滤条件
            distance_metric: 距离度量类型

        Returns:
            搜索结果列表
        """
        if not self.db:
            logger.warning("Database not initialized, no documents to search")
            return []

        # 应用过滤器
        candidate_indices = self._apply_filters(filters)

        # 如果没有候选文档，返回空结果
        if not candidate_indices:
            return []

        # 在候选文档中搜索
        results = []
        query_np = np.array(query_vector)

        for index in candidate_indices:
            if index >= len(self.documents):
                continue

            doc = self.documents[index]

            # 检查是否已删除
            if hasattr(doc, 'meta_data') and doc.meta_data.get('_deleted'):
                continue

            # 检查是否有向量
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            # 计算相似度
            doc_vector = np.array(doc.vector)
            score = self._compute_similarity(query_np, doc_vector, distance_metric)

            results.append(SearchResult(
                document=doc,
                score=score,
                distance=1.0 - score if distance_metric == DistanceMetric.COSINE else None
            ))

        # 按分数排序，返回 top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> List[int]:
        """应用元数据过滤器"""
        if not filters:
            return list(range(len(self.documents)))

        candidate_indices = []

        for index, doc in enumerate(self.documents):
            if not hasattr(doc, 'meta_data') or doc.meta_data is None:
                continue

            # 检查是否满足所有过滤条件
            match = True
            for key, value in filters.items():
                if doc.meta_data.get(key) != value:
                    match = False
                    break

            if match:
                candidate_indices.append(index)

        return candidate_indices

    def _compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: DistanceMetric
    ) -> float:
        """计算向量相似度"""
        if metric == DistanceMetric.COSINE:
            # 余弦相似度
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        elif metric == DistanceMetric.L2:
            # L2 距离（转换为相似度）
            dist = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + dist))

        elif metric == DistanceMetric.INNER_PRODUCT:
            # 内积
            return float(np.dot(vec1, vec2))

        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    def get_document(self, document_id: str) -> Optional[Document]:
        """获取单个文档"""
        if document_id not in self.doc_id_map:
            return None

        index = self.doc_id_map[document_id]
        if index >= len(self.documents):
            return None

        doc = self.documents[index]

        # 检查是否已删除
        if hasattr(doc, 'meta_data') and doc.meta_data.get('_deleted'):
            return None

        return doc

    def get_documents_by_metadata(self, metadata: Dict[str, Any]) -> List[Document]:
        """根据元数据获取文档"""
        results = []

        for doc in self.documents:
            if not hasattr(doc, 'meta_data') or doc.meta_data is None:
                continue

            # 检查是否已删除
            if doc.meta_data.get('_deleted'):
                continue

            # 检查元数据匹配
            match = all(
                doc.meta_data.get(key) == value
                for key, value in metadata.items()
            )

            if match:
                results.append(doc)

        return results

    def update_document(self, document_id: str, document: Document) -> bool:
        """更新文档"""
        if document_id not in self.doc_id_map:
            return False

        index = self.doc_id_map[document_id]
        if index >= len(self.documents):
            return False

        # 保留文档 ID
        if not hasattr(document, 'meta_data') or document.meta_data is None:
            document.meta_data = {}
        document.meta_data['_doc_id'] = document_id

        self.documents[index] = document
        return True

    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """统计文档数量"""
        if filters:
            return len(self.get_documents_by_metadata(filters))

        # 统计未删除的文档
        count = 0
        for doc in self.documents:
            if hasattr(doc, 'meta_data') and doc.meta_data.get('_deleted'):
                continue
            count += 1

        return count

    def save(self, path: str) -> None:
        """
        保存数据库到磁盘

        Args:
            path: 保存路径（.pkl 文件）
        """
        if not self.db:
            logger.warning("No database to save")
            return

        # 如果路径是目录，生成文件名
        if os.path.isdir(path):
            # 从配置中获取仓库名称
            repo_name = self.config.get("repo_name", "database")
            path = os.path.join(path, f"{repo_name}.pkl")

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 使用 AdalFlow 的 save_state
        self.db.save_state(filepath=path)

        # 保存额外的元数据
        metadata_path = path.replace('.pkl', '_metadata.pkl')
        metadata = {
            'doc_id_map': self.doc_id_map,
            'next_doc_id': self.next_doc_id,
            'documents': self.documents
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Database saved to {path}")

    def load(self, path: str) -> None:
        """
        从磁盘加载数据库

        Args:
            path: 加载路径（.pkl 文件）
        """
        if not os.path.exists(path):
            logger.warning(f"Database file not found: {path}")
            return

        # 加载 AdalFlow LocalDB
        self.db = LocalDB.load_state(path)

        # 加载元数据
        metadata_path = path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.doc_id_map = metadata.get('doc_id_map', {})
            self.next_doc_id = metadata.get('next_doc_id', 0)
            self.documents = metadata.get('documents', [])
        else:
            # 如果没有元数据文件，从 db 中加载
            self.documents = self.db.get_transformed_data(key="split_and_embed")
            self.doc_id_map = {}
            self.next_doc_id = len(self.documents)

        logger.info(f"Database loaded from {path}, {len(self.documents)} documents")

    def close(self) -> None:
        """关闭数据库连接"""
        # FAISS 不需要关闭连接
        logger.debug("FAISS backend closed")
