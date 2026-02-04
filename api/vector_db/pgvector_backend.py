"""
PostgreSQL + pgvector 向量数据库后端

使用 PostgreSQL 的 pgvector 扩展实现向量相似度搜索。
"""

import json
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:
    raise ImportError(
        "pgvector backend requires psycopg and pgvector packages. "
        "Install them with: pip install psycopg[binary] pgvector"
    )

from adalflow.core.types import Document

from api.vector_db.base import (
    VectorDBBackend,
    SearchResult,
    DistanceMetric
)


logger = logging.getLogger(__name__)


class PgvectorBackend(VectorDBBackend):
    """
    PostgreSQL + pgvector 向量数据库后端

    支持向量相似度搜索、元数据过滤、事务和并发。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 pgvector 后端

        Args:
            config: 配置字典
                - connection_string: PostgreSQL 连接字符串
                - embedding_dimension: 向量维度
                - index_type: 索引类型（hnsw 或 ivfflat）
                - index_parameters: 索引参数
                - pool_size: 连接池大小
        """
        super().__init__(config)

        self.connection_string = config.get(
            "connection_string",
            "postgresql://deepwiki:deepwiki@localhost:5432/deepwiki"
        )
        self.index_type = config.get("index_type", "hnsw")
        self.index_parameters = config.get("index_parameters", {
            "m": 16,
            "ef_construction": 64
        })

        # 连接池
        self.pool_size = config.get("pool_size", 10)
        self._pool: List[psycopg.Connection] = []
        self._pool_lock = threading.Lock()

        # 当前仓库 ID（需要在 use_repository 中设置）
        self.repository_id: Optional[int] = None

        # 初始化数据库连接
        self._connect()
        self._init_database()

        logger.info(f"pgvector backend initialized (dimension={self.embedding_dimension})")

    def _connect(self):
        """创建连接池"""
        for _ in range(self.pool_size):
            conn = psycopg.connect(self.connection_string)
            # 注册 vector 类型
            register_vector(conn)
            self._pool.append(conn)

        logger.debug(f"Created connection pool with {self.pool_size} connections")

    @contextmanager
    def _get_connection(self):
        """从连接池获取连接"""
        with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                # 如果池为空，创建新连接
                conn = psycopg.connect(self.connection_string)
                register_vector(conn)

        try:
            yield conn
        finally:
            # 归还连接到池
            with self._pool_lock:
                self._pool.append(conn)

    def _init_database(self):
        """初始化数据库（创建表和扩展）"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 启用 pgvector 扩展
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # 创建 repositories 表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS repositories (
                        id SERIAL PRIMARY KEY,
                        owner VARCHAR(255) NOT NULL,
                        repo VARCHAR(255) NOT NULL,
                        repo_url TEXT NOT NULL,
                        current_commit VARCHAR(64),
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(owner, repo)
                    )
                """)

                # 创建 document_chunks 表
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        file_path VARCHAR(1024) NOT NULL,
                        file_type VARCHAR(50),
                        is_code BOOLEAN DEFAULT TRUE,
                        content TEXT NOT NULL,
                        embedding vector({self.embedding_dimension}) NOT NULL,
                        token_count INTEGER,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMP DEFAULT NOW(),
                        deleted BOOLEAN DEFAULT FALSE
                    )
                """)

                # 创建索引
                self._create_indexes(cur)

                conn.commit()

        logger.debug("Database initialized successfully")

    def _create_indexes(self, cursor):
        """创建向量索引和元数据索引"""
        # HNSW 索引（高性能）
        if self.index_type == "hnsw":
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self.index_parameters.get('m', 16)},
                      ef_construction = {self.index_parameters.get('ef_construction', 64)})
            """)
        # IVFFlat 索引（更精确）
        elif self.index_type == "ivfflat":
            lists = self.index_parameters.get("lists", 100)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """)
        # Flat 索引（精确但慢）
        else:
            logger.warning(f"Unknown index type: {self.index_type}, using exact search")

        # 元数据查询索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_repo_idx
            ON document_chunks(repository_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_file_idx
            ON document_chunks(file_path)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx
            ON document_chunks USING gin(metadata)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_deleted_idx
            ON document_chunks(deleted)
        """)

        logger.debug(f"Created {self.index_type} indexes")

    def use_repository(self, owner: str, repo: str, repo_url: str) -> int:
        """
        设置当前使用的仓库

        Args:
            owner: 仓库所有者
            repo: 仓库名称
            repo_url: 仓库 URL

        Returns:
            仓库 ID
        """
        logger.info("=" * 80)
        logger.info(f"[pgvector] use_repository() 被调用")
        logger.info(f"   → owner: {owner}")
        logger.info(f"   → repo: {repo}")
        logger.info(f"   → repo_url: {repo_url}")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 查找或创建仓库
                cur.execute(
                    """
                    INSERT INTO repositories (owner, repo, repo_url)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (owner, repo)
                    DO UPDATE SET repo_url = EXCLUDED.repo_url,
                                  updated_at = NOW()
                    RETURNING id
                    """,
                    (owner, repo, repo_url)
                )
                result = cur.fetchone()
                self.repository_id = result[0]
                conn.commit()

        logger.info(f"[pgvector] ✅ Using repository: {owner}/{repo} (ID={self.repository_id})")
        logger.info("=" * 80)
        return self.repository_id

    def get_current_commit(self) -> Optional[str]:
        """
        获取当前仓库的 commit hash

        Returns:
            Optional[str]: commit hash，如果未设置返回 None
        """
        if self.repository_id is None:
            logger.warning("[pgvector] Repository not set. Call use_repository() first.")
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT current_commit FROM repositories WHERE id = %s",
                        (self.repository_id,)
                    )
                    result = cur.fetchone()
                    if result and result[0]:
                        return result[0]
                    return None
        except Exception as e:
            logger.error(f"[pgvector] Failed to get current commit: {e}")
            return None

    def update_current_commit(self, commit_hash: str) -> bool:
        """
        更新当前仓库的 commit hash

        Args:
            commit_hash: 新的 commit hash

        Returns:
            bool: 是否成功更新
        """
        if self.repository_id is None:
            logger.warning("[pgvector] Repository not set. Call use_repository() first.")
            return False

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE repositories
                        SET current_commit = %s, updated_at = NOW()
                        WHERE id = %s
                        """,
                        (commit_hash, self.repository_id)
                    )
                    conn.commit()
                    logger.info(f"[pgvector] Updated commit to {commit_hash[:8]} for repository_id={self.repository_id}")
                    return True
        except Exception as e:
            logger.error(f"[pgvector] Failed to update current commit: {e}")
            return False

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        批量添加文档

        Args:
            documents: 文档列表

        Returns:
            文档 ID 列表
        """
        logger.info("=" * 80)
        logger.info(f"[pgvector] add_documents() 被调用")
        logger.info(f"   → 文档数: {len(documents)}")
        logger.info(f"   → repository_id: {self.repository_id}")

        if self.repository_id is None:
            logger.error("[pgvector] Repository not set. Call use_repository() first.")
            raise ValueError("Repository not set. Call use_repository() first.")

        # 检查第一个文档的向量状态
        if documents and len(documents) > 0:
            first_doc = documents[0]
            logger.info(f"   → 第一个文档类型: {type(first_doc)}")
            logger.info(f"   → 第一个文档有向量属性: {hasattr(first_doc, 'vector')}")
            if hasattr(first_doc, 'vector'):
                logger.info(f"   → 第一个文档向量值: {first_doc.vector}")
                logger.info(f"   → 第一个文档向量类型: {type(first_doc.vector)}")
                if first_doc.vector is not None and hasattr(first_doc.vector, '__len__'):
                    logger.info(f"   → 第一个文档向量长度: {len(first_doc.vector)}")

        logger.info(f"[pgvector] 正在添加 {len(documents)} 个文档到 repository_id={self.repository_id}")

        document_ids = []
        skipped = 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for idx, doc in enumerate(documents):
                    # 🔍 调试日志：显示每个文档的向量信息
                    logger.info(f"[pgvector] 处理文档 {idx}:")
                    logger.info(f"   → 文件路径: {doc.meta_data.get('file_path', 'unknown')}")
                    logger.info(f"   → 有向量属性: {hasattr(doc, 'vector')}")
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        logger.info(f"   → 向量类型: {type(doc.vector)}")
                        if hasattr(doc.vector, '__len__'):
                            logger.info(f"   → 向量长度: {len(doc.vector)}")

                    # 检查是否有向量
                    if not hasattr(doc, 'vector') or doc.vector is None:
                        logger.warning(f"[pgvector] 文档 {idx} 没有向量，跳过")
                        skipped += 1
                        continue

                    # 🔧 关键修复：验证向量不为空
                    vector = doc.vector
                    if hasattr(vector, '__len__') and len(vector) == 0:
                        logger.error(f"[pgvector] ❌ 文档 {idx} 向量为空，跳过")
                        logger.error(f"   → 文件路径: {doc.meta_data.get('file_path', 'unknown')}")
                        logger.error(f"   → 文本长度: {len(doc.text)} 字符")
                        logger.error(f"   → 向量类型: {type(vector)}")
                        logger.error(f"   → 向量值: {vector}")
                        skipped += 1
                        continue

                    # 🔧 关键修复：验证向量维度
                    vector_length = None
                    if hasattr(vector, '__len__'):
                        vector_length = len(vector)
                    elif hasattr(vector, 'shape') and len(vector.shape) > 0:
                        vector_length = vector.shape[-1]

                    if vector_length is not None and vector_length != self.embedding_dimension:
                        logger.error(f"[pgvector] ❌ 文档 {idx} 向量维度不匹配")
                        logger.error(f"   → 文件路径: {doc.meta_data.get('file_path', 'unknown')}")
                        logger.error(f"   → 期望维度: {self.embedding_dimension}")
                        logger.error(f"   → 实际维度: {vector_length}")
                        logger.error(f"   → 向量类型: {type(vector)}")
                        skipped += 1
                        continue

                    # 🔍 调试日志：显示向量信息（只在第一个文档时）
                    if idx == 0:
                        logger.info(f"[pgvector] 第一个文档向量详情:")
                        logger.info(f"   → 向量类型: {type(doc.vector)}")
                        logger.info(f"   → 向量长度: {vector_length if vector_length else 'unknown'}")
                        logger.info(f"   → 文档元数据: {doc.meta_data}")

                    # 🔧 修复：清理文档内容中的 null 字节（PostgreSQL 不支持）
                    content = doc.text
                    if '\x00' in content:
                        # 替换 null 字节为空格
                        content = content.replace('\x00', ' ')
                        logger.debug(f"[pgvector] 文档 {idx} 包含 null 字节，已清理")

                    # 🔧 关键修复：确保向量是 Python list 格式（不是 numpy array）
                    # pgvector 的 psycopg 适配器会自动处理 list → vector 的转换
                    vector = doc.vector
                    if hasattr(vector, 'tolist'):
                        # numpy array → list
                        vector = vector.tolist()
                    elif not isinstance(vector, list):
                        vector = list(vector)

                    # 🔍 调试日志：显示最终向量信息
                    if idx == 1:
                        logger.info(f"[pgvector] 第二个文档向量详情（插入前）:")
                        logger.info(f"   → 文件路径: {doc.meta_data.get('file_path', 'unknown')}")
                        logger.info(f"   → 向量类型: {type(vector)}")
                        logger.info(f"   → 向量长度: {len(vector) if hasattr(vector, '__len__') else 'unknown'}")
                        logger.info(f"   → 向量前5个值: {vector[:5] if hasattr(vector, '__getitem__') else 'N/A'}")

                    # 插入文档（使用 ::vector 显式类型转换）
                    try:
                        cur.execute("""
                            INSERT INTO document_chunks
                            (repository_id, chunk_index, file_path, file_type,
                             content, embedding, token_count, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                            RETURNING id
                        """, (
                            self.repository_id,
                            doc.meta_data.get('chunk_index', 0),
                            doc.meta_data.get('file_path', ''),
                            doc.meta_data.get('type', ''),
                            content,  # 使用清理后的内容
                            vector,  # Python list，会被 ::vector 转换
                            doc.meta_data.get('token_count', 0),
                            json.dumps(doc.meta_data)
                        ))
                        doc_id = f"pg_{cur.fetchone()[0]}"
                        document_ids.append(doc_id)
                    except Exception as e:
                        logger.error(f"[pgvector] ❌ 文档 {idx} 插入失败")
                        logger.error(f"   → 文件路径: {doc.meta_data.get('file_path', 'unknown')}")
                        logger.error(f"   → 错误类型: {type(e).__name__}")
                        logger.error(f"   → 错误信息: {e}")
                        logger.error(f"   → 向量类型: {type(vector)}")
                        logger.error(f"   → 向量长度: {len(vector) if hasattr(vector, '__len__') else 'unknown'}")
                        logger.error(f"   → 向量前5个值: {vector[:5] if hasattr(vector, '__getitem__') else 'N/A'}")
                        raise

                    if (idx + 1) % 50 == 0:
                        logger.debug(f"[pgvector] 已处理 {idx + 1}/{len(documents)} 个文档")

                # 🔍 调试日志：commit 前的状态
                logger.info(f"[pgvector] 准备 commit")
                logger.info(f"   → 已处理文档数: {len(document_ids)}")
                logger.info(f"   → 跳过文档数: {skipped}")
                logger.info(f"   → 总文档数: {len(documents)}")

                conn.commit()

                logger.info(f"[pgvector] ✅ commit 成功")

        logger.info(f"[pgvector] ✅ 成功添加 {len(document_ids)} 个文档，跳过 {skipped} 个")
        logger.info("=" * 80)
        return document_ids

    def delete_documents(self, document_ids: List[str]) -> int:
        """
        删除文档（软删除）

        Args:
            document_ids: 文档 ID 列表（格式：pg_{id}）

        Returns:
            删除的文档数量
        """
        # 提取数字 ID
        ids = [int(doc_id.split('_')[1]) for doc_id in document_ids if '_' in doc_id]

        if not ids:
            return 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE document_chunks SET deleted = TRUE WHERE id = ANY(%s)",
                    (ids,)
                )
                deleted_count = cur.rowcount
                conn.commit()

        logger.info(f"Deleted {deleted_count} documents")
        return deleted_count

    def delete_by_metadata(self, key: str, value: Any) -> int:
        """根据元数据删除文档"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE document_chunks
                    SET deleted = TRUE
                    WHERE repository_id = %s
                      AND metadata->>%s = %s
                      AND deleted = FALSE
                    """,
                    (self.repository_id, key, str(value))
                )
                deleted_count = cur.rowcount
                conn.commit()

        return deleted_count

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
        if self.repository_id is None:
            raise ValueError("Repository not set. Call use_repository() first.")

        # 根据度量类型选择运算符
        if distance_metric == DistanceMetric.COSINE:
            operator = "<=>"  # 余弦距离
        elif distance_metric == DistanceMetric.L2:
            operator = "<->"  # L2 距离
        elif distance_metric == DistanceMetric.INNER_PRODUCT:
            operator = "<#>"  # 内积（负距离）
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # 🔧 关键修复：确保向量是 Python list 格式
        vector = query_vector
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        elif not isinstance(vector, list):
            vector = list(vector)

        # 🔧 关键修复：使用 %s::vector 显式类型转换
        # register_vector() 注册了类型，但 SQL 中仍需显式转换
        sql = f"""
            SELECT id, file_path, content, metadata,
                   1 - (embedding {operator} %s::vector) as similarity
            FROM document_chunks
            WHERE repository_id = %s
              AND deleted = FALSE
        """

        params = [vector, self.repository_id]

        # 添加过滤条件
        if filters:
            for key, value in filters.items():
                if key == 'file_path':
                    sql += " AND file_path = %s"
                    params.append(value)
                elif key == 'is_code':
                    sql += " AND is_code = %s"
                    params.append(value)
                else:
                    # JSONB 元数据查询
                    sql += " AND metadata->>%s = %s"
                    params.extend([key, str(value)])

        # 排序并限制结果（同样需要 ::vector 类型转换）
        sql += f" ORDER BY embedding {operator} %s::vector LIMIT %s"
        params.extend([vector, top_k])

        # 执行查询
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results = cur.fetchall()

        # 转换为 SearchResult 对象
        search_results = []
        for row in results:
            doc = Document(
                text=row[2],
                meta_data=row[3]
            )
            search_results.append(SearchResult(
                document=doc,
                score=float(row[4]),  # 相似度（0-1）
                distance=None
            ))

        logger.debug(f"Found {len(search_results)} results")
        return search_results

    def get_document(self, document_id: str) -> Optional[Document]:
        """获取单个文档"""
        if '_' not in document_id:
            return None

        doc_id = int(document_id.split('_')[1])

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, metadata, vector
                    FROM document_chunks
                    WHERE id = %s AND deleted = FALSE
                    """,
                    (doc_id,)
                )
                result = cur.fetchone()

                if not result:
                    return None

                # 从文本和元数据重建 Document
                # 注意：向量无法从 PostgreSQL 直接恢复为 numpy 数组
                # 如果需要向量，需要使用 embedding_to_array 函数
                return Document(
                    text=result[0],
                    meta_data=result[1]
                )

    def get_documents_by_metadata(self, metadata: Dict[str, Any]) -> List[Document]:
        """根据元数据获取文档"""
        sql = """
            SELECT content, metadata
            FROM document_chunks
            WHERE repository_id = %s AND deleted = FALSE
        """
        params = [self.repository_id]

        for key, value in metadata.items():
            sql += " AND metadata->>%s = %s"
            params.extend([key, str(value)])

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results = cur.fetchall()

        return [
            Document(text=row[0], meta_data=row[1])
            for row in results
        ]

    def update_document(self, document_id: str, document: Document) -> bool:
        """更新文档"""
        if '_' not in document_id:
            return False

        doc_id = int(document_id.split('_')[1])

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 更新内容和元数据
                update_fields = ["content = %s", "metadata = %s"]
                params = [document.text, json.dumps(document.meta_data)]

                # 🔧 关键修复：如果有向量，也更新（使用 ::vector 类型转换）
                if hasattr(document, 'vector') and document.vector is not None:
                    update_fields.append("embedding = %s::vector")
                    vector = document.vector
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    elif not isinstance(vector, list):
                        vector = list(vector)
                    params.append(vector)

                params.append(doc_id)

                cur.execute(f"""
                    UPDATE document_chunks
                    SET {', '.join(update_fields)}
                    WHERE id = %s AND deleted = FALSE
                """, params)

                success = cur.rowcount > 0
                conn.commit()

        return success

    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """统计文档数量"""
        sql = "SELECT COUNT(*) FROM document_chunks WHERE repository_id = %s AND deleted = FALSE"
        params = [self.repository_id]

        if filters:
            for key, value in filters.items():
                sql += " AND metadata->>%s = %s"
                params.extend([key, str(value)])

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                count = cur.fetchone()[0]

        return count

    def save(self, path: str) -> None:
        """
        保存数据库（PostgreSQL 自动持久化）

        这个方法是 NOP，因为 PostgreSQL 自动持久化数据。
        """
        logger.debug("PostgreSQL auto-persists data, save() is a NOP")

    def load(self, path: str) -> None:
        """
        加载数据库（PostgreSQL 自动持久化）

        这个方法是 NOP，因为数据已经在数据库中。
        """
        logger.debug("PostgreSQL data is already loaded, load() is a NOP")

    def close(self) -> None:
        """关闭所有数据库连接"""
        with self._pool_lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()

        logger.debug("Closed all database connections")


# 导入 threading（用于连接池锁）
import threading
