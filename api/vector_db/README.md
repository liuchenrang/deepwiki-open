# 向量数据库抽象层

支持多种向量数据库后端的统一接口，使 DeepWiki 可以灵活选择向量存储方案。

## 支持的后端

| 后端 | 说明 | 状态 |
|------|------|------|
| **FAISS** | 本地向量数据库（默认） | ✅ 稳定 |
| **pgvector** | PostgreSQL + pgvector 扩展 | ✅ 稳定 |

## 快速开始

### 1. 安装依赖

**FAISS（默认）：**
```bash
pip install faiss-cpu
```

**pgvector：**
```bash
pip install psycopg[binary] pgvector
```

### 2. 配置后端

**方法 A：环境变量**
```bash
# 使用 FAISS（默认）
export VECTOR_DB_BACKEND=faiss
export FAISS_INDEX_PATH=~/.adalflow/databases/

# 使用 pgvector
export VECTOR_DB_BACKEND=pgvector
export DATABASE_URL=postgresql://deepwiki:deepwiki@localhost:5432/deepwiki
```

**方法 B：配置文件**
```bash
cp api/config/db.json.example api/config/db.json
# 编辑 api/config/db.json
```

### 3. 初始化 pgvector 数据库（如果使用 pgvector）

```bash
# 启动 PostgreSQL（Docker）
docker-compose up -d postgres

# 初始化数据库
python -m api.tools.init_pgvector_db init

# 测试连接
python -m api.tools.init_pgvector_db test
```

### 4. 使用示例

```python
from api.vector_db import get_vector_db, VectorDBConfig

# 方法 1：从环境变量加载配置
db = get_vector_db()

# 方法 2：从字典加载配置
config = {
    "backend": "faiss",
    "faiss": {
        "index_path": "~/.adalflow/databases/",
        "embedding_dimension": 1024
    }
}
db = get_vector_db(config)

# 使用数据库（统一接口）
db.add_documents(documents)
results = db.search(query_vector, top_k=20)
db.delete_documents(doc_ids)
db.close()
```

## Docker 部署

### 使用 Docker Compose

```bash
# 启动所有服务（包括 PostgreSQL）
docker-compose up -d

# 查看日志
docker-compose logs -f postgres

# 初始化数据库
docker-compose exec deepwiki python -m api.tools.init_pgvector_db init
```

### 环境变量配置

在 `.env` 文件中添加：

```bash
# 向量数据库配置
VECTOR_DB_BACKEND=pgvector
DATABASE_URL=postgresql://deepwiki:deepwiki@postgres:5432/deepwiki
EMBEDDING_DIMENSION=1024
PGVECTOR_INDEX_TYPE=hnsw

# PostgreSQL 配置
POSTGRES_USER=deepwiki
POSTGRES_PASSWORD=deepwiki
POSTGRES_DB=deepwiki
POSTGRES_PORT=5432
```

## 测试

### 运行测试

```bash
# 测试 FAISS 后端
python -m api.vector_db.tests.test_vector_db

# 测试 pgvector 后端（需要先启动 PostgreSQL）
export DATABASE_URL=postgresql://deepwiki:deepwiki@localhost:5432/deepwiki
python -m api.vector_db.tests.test_vector_db
```

### 测试覆盖率

```bash
pip install pytest pytest-cov
pytest api/vector_db/tests/ --cov=api/vector_db --cov-report=html
```

## API 参考

### VectorDBBackend

抽象基类，定义了所有向量数据库后端必须实现的接口。

**方法：**

- `add_documents(documents: List[Document]) -> List[str]`
- `delete_documents(document_ids: List[str]) -> int`
- `delete_by_metadata(key: str, value: Any) -> int`
- `search(query_vector: List[float], top_k: int, filters: Dict, distance_metric: DistanceMetric) -> List[SearchResult]`
- `get_document(document_id: str) -> Optional[Document]`
- `get_documents_by_metadata(metadata: Dict) -> List[Document]`
- `update_document(document_id: str, document: Document) -> bool`
- `count_documents(filters: Dict) -> int`
- `save(path: str) -> None`
- `load(path: str) -> None`
- `close() -> None`

### 工厂方法

**`get_vector_db(config: Any) -> VectorDBBackend`**

创建向量数据库实例。

**参数：**
- `config`: 配置对象，可以是：
  - `None`：从环境变量加载
  - `dict`: 配置字典
  - `VectorDBConfig`: 配置对象

**返回：**
- `VectorDBBackend` 实例

## 性能对比

| 指标 | FAISS | pgvector |
|------|-------|----------|
| 单机查询速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 并发支持 | ⚠️ 需要自己实现 | ⭐⭐⭐⭐⭐ |
| 增量更新 | ⚠️ 困难 | ⭐⭐⭐⭐⭐ |
| 元数据过滤 | ⚠️ 需要自己维护 | ⭐⭐⭐⭐⭐ |
| 扩展性 | ⚠️ 单机 | ⭐⭐⭐⭐⭐ |
| 部署复杂度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**建议：**
- 小型项目（<1000 文件）：FAISS
- 中大型项目（>1000 文件）：pgvector
- 需要增量更新：pgvector
- 多用户并发：pgvector

## 迁移指南

### 从 FAISS 迁移到 pgvector

1. **安装依赖**
   ```bash
   pip install psycopg[binary] pgvector
   ```

2. **启动 PostgreSQL**
   ```bash
   docker-compose up -d postgres
   ```

3. **初始化数据库**
   ```bash
   python -m api.tools.init_pgvector_db init
   ```

4. **切换配置**
   ```bash
   export VECTOR_DB_BACKEND=pgvector
   export DATABASE_URL=postgresql://deepwiki:deepwiki@localhost:5432/deepwiki
   ```

5. **迁移数据**（可选）
   ```python
   from api.tools.migrate_to_pgvector import migrate_faiss_to_pgvector

   migrate_faiss_to_pgvector(
       pkl_path="~/.adalflow/databases/owner_repo.pkl",
       repo_url="https://github.com/owner/repo"
   )
   ```

## 故障排除

### 问题 1：pgvector 扩展未安装

**错误：** `extension "vector" does not exist`

**解决：**
```bash
# 检查 pgvector 是否安装
docker-compose exec postgres psql -U deepwiki -d deepwiki -c "CREATE EXTENSION vector;"
```

### 问题 2：连接被拒绝

**错误：** `connection refused` 或 `could not connect to server`

**解决：**
```bash
# 检查 PostgreSQL 是否运行
docker-compose ps postgres

# 查看 PostgreSQL 日志
docker-compose logs postgres

# 重启 PostgreSQL
docker-compose restart postgres
```

### 问题 3：向量维度不匹配

**错误：** `vector must have 1024 dimensions`（实际是 768）

**解决：**
```bash
# 设置正确的向量维度
export EMBEDDING_DIMENSION=768
```

## 开发指南

### 添加新的向量数据库后端

1. **创建后端类**

```python
# api/vector_db/my_backend.py
from api.vector_db.base import VectorDBBackend

class MyBackend(VectorDBBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 初始化你的数据库

    def add_documents(self, documents: List[Document]) -> List[str]:
        # 实现添加文档逻辑
        pass

    # 实现其他必需方法...
```

2. **注册后端**

```python
# api/vector_db/__init__.py
from .my_backend import MyBackend
__all__.append("MyBackend")
```

3. **更新工厂方法**

```python
# api/vector_db/factory.py
def get_vector_db(config: Any) -> VectorDBBackend:
    # ...
    elif backend == "my_backend":
        from api.vector_db.my_backend import MyBackend
        return MyBackend(vector_db_config.config)
```

## 贡献指南

欢迎贡献！请查看项目根目录的 `CONTRIBUTING.md`。

## 许可证

MIT License - 详见项目根目录的 `LICENSE` 文件。
