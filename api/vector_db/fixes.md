# 向量数据库问题修复方案

## 问题总结

### 1. 文件丢失问题
- **表现**：某些文件（如 `ywmall/bz/grade.php`）在向量库中缺失
- **影响**：无法检索到相关代码，回答不完整

### 2. 文件重复问题
- **表现**：同一文件存在多个向量版本（如多个 `index.php`）
- **影响**：历史代码向量干扰搜索结果，降低回答准确性

---

## 根本原因

### 重复问题核心原因

**数据库表缺少唯一约束 + 增量更新逻辑缺陷**

```sql
-- 当前表结构（有缺陷）
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    -- ❌ 缺少：(repository_id, file_path, chunk_index) 的唯一约束
    ...
)
```

增量更新流程：
```
文件修改 → 重新生成向量 → ❌ 直接 INSERT（保留旧向量）
                              → 搜索时返回多个版本 → ❌ 回答混乱
```

---

## 解决方案

### 方案 A：完整性约束 + UPSERT（推荐）

#### 1. 修改表结构

```sql
-- 添加唯一约束
ALTER TABLE document_chunks
DROP CONSTRAINT IF EXISTS document_chunks_unique_file;

ALTER TABLE document_chunks
ADD CONSTRAINT document_chunks_unique_file
UNIQUE (repository_id, file_path, chunk_index)
WHERE deleted = FALSE;
```

#### 2. 修改 add_documents 方法

在 `api/vector_db/pgvector_backend.py` 中修改：

```python
def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
    """批量添加文档（使用 UPSERT 避免重复）"""

    if self.repository_id is None:
        raise ValueError("Repository not set. Call use_repository() first.")

    document_ids = []
    skipped = 0
    updated = 0  # 新增：记录更新的文档数

    with self._get_connection() as conn:
        with conn.cursor() as cur:
            for idx, doc in enumerate(documents):
                if not hasattr(doc, 'vector') or doc.vector is None:
                    logger.warning(f"Document {idx} has no vector, skipping")
                    skipped += 1
                    continue

                vector_str = str(doc.vector)
                file_path = doc.meta_data.get('file_path', '')
                chunk_index = doc.meta_data.get('chunk_index', 0)

                # ✅ 使用 INSERT ... ON CONFLICT (UPSERT)
                cur.execute("""
                    INSERT INTO document_chunks
                    (repository_id, chunk_index, file_path, file_type,
                     content, embedding, token_count, metadata, deleted)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, FALSE)
                    ON CONFLICT (repository_id, file_path, chunk_index)
                    WHERE deleted = FALSE
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW(),
                        deleted = FALSE
                    RETURNING id
                """, (
                    self.repository_id,
                    chunk_index,
                    file_path,
                    doc.meta_data.get('type', ''),
                    doc.text,
                    vector_str,
                    doc.meta_data.get('token_count', 0),
                    json.dumps(doc.meta_data)
                ))

                doc_id = f"pg_{cur.fetchone()[0]}"
                document_ids.append(doc_id)
                updated += 1

                if (idx + 1) % 50 == 0:
                    logger.debug(f"Processed {idx + 1}/{len(documents)} documents")

            conn.commit()

    logger.info(f"✅ Added/Updated {len(document_ids)} documents (skipped {skipped})")
    return document_ids
```

#### 3. 增强文件读取日志

在 `api/data_pipeline.py` 的 `read_all_documents` 方法中添加：

```python
# 在文件读取循环后添加
logger.info(f"📊 文件读取统计:")
logger.info(f"   → 成功读取: {len(documents)} 个文件")
logger.info(f"   → 跳过的大文件: {large_file_count}")
logger.info(f"   → 读取失败的文件: {failed_files}")

if failed_files:
    logger.warning(f"⚠️ 以下文件读取失败:")
    for file_path, error in failed_files[:10]:  # 只显示前10个
        logger.warning(f"   - {file_path}: {error}")
```

#### 4. 添加数据验证工具

创建 `api/vector_db/validate.py`：

```python
"""向量数据库数据验证工具"""

import logging
from typing import Dict, List
from api.vector_db import get_vector_db

logger = logging.getLogger(__name__)


def validate_repository_data(owner: str, repo: str) -> Dict[str, any]:
    """
    验证仓库的向量数据完整性

    Returns:
        dict: {
            "total_files": 总文件数,
            "duplicated_files": 重复文件列表,
            "missing_files": 缺失文件列表,
            "health_score": 健康分数 (0-100)
        }
    """
    vector_db = get_vector_db()
    vector_db.use_repository(owner=owner, repo=repo)

    with vector_db._get_connection() as conn:
        with conn.cursor() as cur:
            # 检查重复文件
            cur.execute("""
                SELECT file_path, COUNT(*) as count
                FROM document_chunks
                WHERE repository_id = %s AND deleted = FALSE
                GROUP BY file_path
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """, (vector_db.repository_id,))

            duplicates = {row[0]: row[1] for row in cur.fetchall()}

            # 获取所有文件列表
            cur.execute("""
                SELECT DISTINCT file_path
                FROM document_chunks
                WHERE repository_id = %s AND deleted = FALSE
            """, (vector_db.repository_id,))

            files_in_db = {row[0] for row in cur.fetchall()}

    return {
        "total_files": len(files_in_db),
        "duplicated_files": duplicates,
        "health_score": 100 - len(duplicates) * 5
    }


def clean_duplicates(owner: str, repo: str, dry_run: bool = True) -> int:
    """
    清理重复的向量数据

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        dry_run: 是否只模拟不执行

    Returns:
        int: 清理的文档数量
    """
    vector_db = get_vector_db()
    vector_db.use_repository(owner=owner, repo=repo)

    cleaned = 0

    with vector_db._get_connection() as conn:
        with conn.cursor() as cur:
            # 查找重复文档（保留最新的）
            cur.execute("""
                DELETE FROM document_chunks
                WHERE id IN (
                    SELECT id
                    FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY repository_id, file_path, chunk_index
                                   ORDER BY created_at DESC
                               ) as rn
                        FROM document_chunks
                        WHERE repository_id = %s AND deleted = FALSE
                    ) sub
                    WHERE rn > 1
                )
                RETURNING file_path
            """, (vector_db.repository_id,))

            deleted_files = cur.fetchall()

            if not dry_run:
                conn.commit()
                cleaned = len(deleted_files)
            else:
                cleaned = len(deleted_files)

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaned {cleaned} duplicate documents")
    return cleaned
```

---

### 方案 B：先删除后添加（简单但有缺陷）

```python
def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
    """先删除旧向量，再添加新向量"""

    # ⚠️ 问题：事务期间搜索会返回空结果
    with self._get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                file_path = doc.meta_data.get('file_path', '')

                # 先删除该文件的所有向量
                cur.execute("""
                    DELETE FROM document_chunks
                    WHERE repository_id = %s AND file_path = %s
                """, (self.repository_id, file_path))

                # 再添加新向量
                cur.execute("""
                    INSERT INTO document_chunks ...
                """, (...))
```

**缺点**：在删除和添加之间，搜索会返回空结果

---

### 方案 C：全量重建（彻底但耗时）

```bash
# 手动清理并重建
docker-compose exec deepwiki python -c "
from api.vector_db import get_vector_db

vector_db = get_vector_db()
vector_db.use_repository(owner='local', repo='ywmall')

# 清空所有数据
with vector_db._get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('DELETE FROM document_chunks WHERE repository_id = %s',
                   (vector_db.repository_id,))
        conn.commit()

print('✅ 数据已清空，请重新刷新 Wiki')
"
```

---

## 实施步骤

### 第一步：紧急修复（解决重复问题）

1. **添加唯一约束**
```bash
docker-compose exec db psql -U deepwiki -d deepwiki -c "
ALTER TABLE document_chunks
ADD CONSTRAINT document_chunks_unique_file
UNIQUE (repository_id, file_path, chunk_index)
WHERE deleted = FALSE;
"
```

2. **清理现有重复**
```bash
docker-compose exec deepwiki python -c "
from api.vector_db.validate import clean_duplicates
clean_duplicates(owner='local', repo='ywmall', dry_run=False)
"
```

### 第二步：代码修复（避免新重复）

修改 `api/vector_db/pgvector_backend.py` 的 `add_documents` 方法（见方案 A）

### 第三步：诊断丢失文件

```bash
# 检查哪些文件缺失
docker-compose exec deepwiki python -c "
import os
from pathlib import Path

# 扫描本地文件
local_files = set()
repo_path = '/app/.adalflow/repos/local_ywmall'
for ext in ['.php', '.js', '.py']:
    local_files.update(Path(repo_path).rglob(f'*{ext}'))

# 对比数据库
from api.vector_db import get_vector_db
vector_db = get_vector_db()
vector_db.use_repository(owner='local', repo='ywmall')

with vector_db._get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('''
            SELECT DISTINCT file_path
            FROM document_chunks
            WHERE repository_id = %s AND deleted = FALSE
        ''', (vector_db.repository_id,))
        db_files = {row[0] for row in cur.fetchall()}

missing = local_files - db_files
print(f'缺失文件数: {len(missing)}')
for f in list(missing)[:20]:
    print(f'  - {f}')
"
```

### 第四步：增强日志（预防未来问题）

在 `api/data_pipeline.py` 中添加详细日志，记录：
- 每个文件的处理状态
- 跳过的文件及原因
- 失败的文件及错误信息

---

## 预期效果

### 修复前
```
搜索 "index.php" → 返回 10 个结果
  - index.php (v1, 3天前) ❌
  - index.php (v2, 2天前) ❌
  - index.php (v3, 1天前) ❌
  - ... (7个历史版本)
  → 回答混淆，不准确

搜索 "grade.php" → 返回 0 个结果
  → 文件丢失，无法回答
```

### 修复后
```
搜索 "index.php" → 返回 3 个结果
  - admin/index.php (最新版本) ✅
  - api/index.php (最新版本) ✅
  - public/index.php (最新版本) ✅
  → 回答准确，基于最新代码

搜索 "grade.php" → 返回 1 个结果
  - ywmall/bz/grade.php ✅
  → 完整回答
```

---

## 监控建议

### 1. 定期数据健康检查
```python
# 添加到管理后台的监控任务
def check_database_health():
    """每天检查一次数据完整性"""
    repos = get_all_repositories()
    for owner, repo in repos:
        report = validate_repository_data(owner, repo)
        if report['health_score'] < 90:
            send_alert(f"Repository {owner}/{repo} health score: {report['health_score']}")
```

### 2. Wiki 刷新后验证
```python
# 在 prepare_db_index 结束时添加
def prepare_db_index(...):
    # ... 现有逻辑 ...

    # ✅ 添加验证步骤
    if self.use_vector_db:
        validate_data_integrity(self.vector_db, documents)
```

---

## 总结

| 问题 | 根本原因 | 解决方案 | 优先级 |
|------|---------|---------|--------|
| 文件重复 | 表缺少唯一约束 + 增量更新直接 INSERT | 添加约束 + UPSERT | 🔴 高 |
| 文件丢失 | 读取/向量化失败 + 缺少错误处理 | 增强日志 + 验证工具 | 🟡 中 |
| 回答混乱 | 历史向量干扰搜索 | UPSERT 确保唯一性 | 🔴 高 |

**推荐实施顺序**：
1. 🔴 立即：添加数据库唯一约束 + 清理重复
2. 🟡 本周：修改代码使用 UPSERT
3. 🟢 下周：增强日志和监控
