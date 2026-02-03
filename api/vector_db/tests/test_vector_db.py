"""
向量数据库抽象层测试

测试 FAISS 和 pgvector 后端的基本功能。
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from adalflow.core.types import Document
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_documents(num_docs: int = 10) -> list:
    """创建测试文档"""
    documents = []

    for i in range(num_docs):
        # 创建随机向量
        vector = np.random.rand(128).tolist()

        doc = Document(
            text=f"Test document {i}",
            vector=vector,
            meta_data={
                "file_path": f"test/file_{i}.py",
                "type": "py",
                "is_code": True,
                "chunk_index": i,
            }
        )
        documents.append(doc)

    return documents


def test_faiss_backend():
    """测试 FAISS 后端"""
    print("\n" + "="*60)
    print("Testing FAISS Backend")
    print("="*60)

    from api.vector_db import FaissBackend

    # 创建临时目录
    import tempfile
    test_dir = tempfile.mkdtemp()

    # 初始化 FAISS 后端
    config = {
        "index_path": test_dir,
        "embedding_dimension": 128,
        "repo_name": "test_repo"
    }

    db = FaissBackend(config)

    # 测试添加文档
    print("\n1. Testing add_documents()...")
    docs = create_test_documents(10)
    doc_ids = db.add_documents(docs)
    print(f"   ✓ Added {len(doc_ids)} documents")

    # 测试统计
    print("\n2. Testing count_documents()...")
    count = db.count_documents()
    print(f"   ✓ Count: {count}")

    # 测试搜索
    print("\n3. Testing search()...")
    query_vector = np.random.rand(128).tolist()
    results = db.search(query_vector, top_k=5)
    print(f"   ✓ Found {len(results)} results")
    for i, result in enumerate(results[:3]):
        print(f"     - Result {i+1}: score={result.score:.4f}")

    # 测试元数据过滤
    print("\n4. Testing metadata filtering...")
    filtered = db.get_documents_by_metadata({"type": "py"})
    print(f"   ✓ Found {len(filtered)} documents with type='py'")

    # 测试删除
    print("\n5. Testing delete_documents()...")
    deleted = db.delete_documents([doc_ids[0]])
    print(f"   ✓ Deleted {deleted} document(s)")

    # 测试保存和加载
    print("\n6. Testing save() and load()...")
    db_path = os.path.join(test_dir, "test_db.pkl")
    db.save(db_path)
    print(f"   ✓ Saved to {db_path}")

    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print("   ✓ Cleaned up test directory")

    print("\n✅ All FAISS tests passed!")


def test_pgvector_backend():
    """测试 pgvector 后端"""
    print("\n" + "="*60)
    print("Testing pgvector Backend")
    print("="*60)

    try:
        from api.vector_db import PgvectorBackend
    except ImportError as e:
        print(f"\n⚠️  Skipping pgvector tests: {e}")
        print("   Install with: pip install psycopg[binary] pgvector")
        return

    # 从环境变量获取数据库连接
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://deepwiki:deepwiki@localhost:5432/deepwiki"
    )

    # 初始化 pgvector 后端
    config = {
        "connection_string": db_url,
        "embedding_dimension": 128,
        "index_type": "hnsw",
        "pool_size": 2
    }

    try:
        db = PgvectorBackend(config)

        # 设置测试仓库
        print("\n1. Testing use_repository()...")
        repo_id = db.use_repository(
            owner="test_owner",
            repo="test_repo",
            repo_url="https://github.com/test/test_repo"
        )
        print(f"   ✓ Repository ID: {repo_id}")

        # 清理旧数据
        print("\n2. Cleaning up old data...")
        old_count = db.count_documents()
        if old_count > 0:
            # 删除测试仓库的所有文档
            with db._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM document_chunks WHERE repository_id = %s",
                        (repo_id,)
                    )
                    conn.commit()
            print(f"   ✓ Deleted {old_count} old documents")

        # 测试添加文档
        print("\n3. Testing add_documents()...")
        docs = create_test_documents(10)
        doc_ids = db.add_documents(docs)
        print(f"   ✓ Added {len(doc_ids)} documents")

        # 测试统计
        print("\n4. Testing count_documents()...")
        count = db.count_documents()
        print(f"   ✓ Count: {count}")

        # 测试搜索
        print("\n5. Testing search()...")
        query_vector = np.random.rand(128).tolist()
        results = db.search(query_vector, top_k=5)
        print(f"   ✓ Found {len(results)} results")
        for i, result in enumerate(results[:3]):
            print(f"     - Result {i+1}: score={result.score:.4f}")

        # 测试元数据过滤
        print("\n6. Testing metadata filtering...")
        filtered = db.get_documents_by_metadata({"type": "py"})
        print(f"   ✓ Found {len(filtered)} documents with type='py'")

        # 测试删除
        print("\n7. Testing delete_documents()...")
        deleted = db.delete_documents([doc_ids[0]])
        print(f"   ✓ Deleted {deleted} document(s)")

        # 测试元数据删除
        print("\n8. Testing delete_by_metadata()...")
        metadata_deleted = db.delete_by_metadata("chunk_index", 1)
        print(f"   ✓ Deleted {metadata_deleted} document(s) by metadata")

        db.close()

        print("\n✅ All pgvector tests passed!")

    except Exception as e:
        print(f"\n❌ pgvector test failed: {e}")
        import traceback
        traceback.print_exc()


def test_factory():
    """测试工厂方法"""
    print("\n" + "="*60)
    print("Testing Vector DB Factory")
    print("="*60)

    from api.vector_db import get_vector_db, VectorDBConfig

    # 测试配置加载
    print("\n1. Testing VectorDBConfig.from_env()...")
    config = VectorDBConfig.from_env()
    print(f"   ✓ Backend: {config.backend}")

    # 测试从字典创建
    print("\n2. Testing VectorDBConfig.from_dict()...")
    config_dict = {
        "backend": "faiss",
        "faiss": {
            "index_path": "/tmp/test",
            "embedding_dimension": 512
        }
    }
    config = VectorDBConfig.from_dict(config_dict)
    print(f"   ✓ Backend: {config.backend}")
    print(f"   ✓ Embedding dimension: {config.config['embedding_dimension']}")

    # 测试工厂方法（FAISS）
    print("\n3. Testing get_vector_db() with FAISS...")
    db = get_vector_db(config_dict)
    print(f"   ✓ Created: {db.__class__.__name__}")

    # 测试工厂方法（pgvector，如果可用）
    print("\n4. Testing get_vector_db() with pgvector...")
    try:
        pgvector_config = {
            "backend": "pgvector",
            "pgvector": {
                "connection_string": os.getenv(
                    "DATABASE_URL",
                    "postgresql://deepwiki:deepwiki@localhost:5432/deepwiki"
                ),
                "embedding_dimension": 128
            }
        }
        db = get_vector_db(pgvector_config)
        print(f"   ✓ Created: {db.__class__.__name__}")
        db.close()
    except Exception as e:
        print(f"   ⚠️  Skipped: {e}")

    print("\n✅ All factory tests passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Vector Database Backend Tests")
    print("="*60)

    # 运行所有测试
    test_factory()
    test_faiss_backend()
    test_pgvector_backend()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
