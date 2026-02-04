"""Git 工具模块 - 用于检测和处理代码变更

提供基于 git 的增量更新功能，包括：
1. 检测文件变更（新增、修改、删除）
2. 获取当前 commit hash
3. 比较两个版本之间的差异
4. 自动初始化 git 仓库（对于非仓库目录）
"""

import os
import subprocess
import logging
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class FileChangeType(Enum):
    """文件变更类型"""
    ADDED = "added"      # 新增文件
    MODIFIED = "modified" # 修改文件
    DELETED = "deleted"   # 删除文件
    RENAMED = "renamed"   # 重命名文件


@dataclass
class FileChange:
    """文件变更信息"""
    change_type: FileChangeType
    file_path: str      # 相对于仓库根目录的路径
    old_path: Optional[str] = None  # 对于重命名文件，记录旧路径


@dataclass
class GitChanges:
    """Git 变更信息"""
    commit_hash: str           # 当前 commit hash
    previous_hash: Optional[str] # 上一个 commit hash（首次为 None）
    changes: List[FileChange]   # 变更列表
    is_initial: bool           # 是否首次提交（无历史）


def run_git_command(repo_path: str, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    在指定仓库目录执行 git 命令

    Args:
        repo_path: 仓库路径
        args: git 命令参数（不包括 'git'）
        check: 是否检查返回码

    Returns:
        subprocess.CompletedProcess: 命令执行结果
    """
    cmd = ["git", "-C", repo_path] + args
    logger.debug(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check
    )

    return result


def is_git_repository(repo_path: str) -> bool:
    """
    检查目录是否是 git 仓库

    Args:
        repo_path: 目录路径

    Returns:
        bool: 是否是 git 仓库
    """
    git_dir = os.path.join(repo_path, ".git")
    return os.path.exists(git_dir)


def initialize_git_repo(repo_path: str) -> bool:
    """
    初始化 git 仓库（对于非仓库目录）

    Args:
        repo_path: 目录路径

    Returns:
        bool: 是否成功初始化
    """
    try:
        if is_git_repository(repo_path):
            logger.info(f"Path {repo_path} is already a git repository")
            return True

        logger.info(f"Initializing git repository at {repo_path}")
        result = run_git_command(repo_path, ["init"], check=True)

        # 配置用户信息（如果未配置）
        try:
            run_git_command(repo_path, ["config", "user.email", "deepwiki@localhost"], check=False)
            run_git_command(repo_path, ["config", "user.name", "DeepWiki"], check=False)
        except:
            pass  # 配置失败不影响使用

        # 创建初始提交
        run_git_command(repo_path, ["add", "."], check=False)
        try:
            run_git_command(repo_path, ["commit", "-m", "Initial commit by DeepWiki"], check=False)
        except:
            # 可能没有可提交的内容，忽略错误
            pass

        logger.info(f"Successfully initialized git repository at {repo_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize git repository at {repo_path}: {e}")
        return False


def get_current_commit(repo_path: str) -> Optional[str]:
    """
    获取当前 commit hash

    Args:
        repo_path: 仓库路径

    Returns:
        Optional[str]: commit hash，如果失败返回 None
    """
    try:
        result = run_git_command(repo_path, ["rev-parse", "HEAD"], check=False)
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            logger.debug(f"Current commit: {commit_hash}")
            return commit_hash
        return None
    except Exception as e:
        logger.error(f"Failed to get current commit: {e}")
        return None


def get_previous_commit(repo_path: str) -> Optional[str]:
    """
    获取上一个 commit hash

    Args:
        repo_path: 仓库路径

    Returns:
        Optional[str]: 上一个 commit hash，如果不存在返回 None
    """
    try:
        result = run_git_command(repo_path, ["rev-parse", "HEAD~1"], check=False)
        if result.returncode == 0:
            previous_hash = result.stdout.strip()
            logger.debug(f"Previous commit: {previous_hash}")
            return previous_hash
        return None
    except Exception as e:
        logger.debug(f"No previous commit found: {e}")
        return None


def commit_uncommitted_changes(repo_path: str) -> bool:
    """
    自动提交未提交的更改（用于本地目录的增量刷新）

    Args:
        repo_path: 仓库路径

    Returns:
        bool: 是否成功提交（或无需提交）
    """
    try:
        # 检查是否有未提交的更改
        result = run_git_command(repo_path, ["status", "--porcelain"], check=False)
        if result.returncode != 0:
            logger.debug(f"Failed to check git status: {result.stderr}")
            return True  # 继续执行，不是致命错误

        uncommitted = result.stdout.strip()
        if not uncommitted:
            logger.debug(f"No uncommitted changes in {repo_path}")
            return True

        logger.info(f"📝 Detected uncommitted changes in {repo_path}")
        logger.debug(f"Uncommitted files:\n{uncommitted}")

        # 添加所有更改
        run_git_command(repo_path, ["add", "."], check=False)
        logger.debug(f"Staged all changes")

        # 提交更改
        result = run_git_command(
            repo_path,
            ["commit", "-m", "Auto commit by DeepWiki"],
            check=False
        )

        if result.returncode == 0:
            logger.info(f"✅ Successfully committed uncommitted changes")
            return True
        else:
            # 可能是因为没有可提交的内容（所有文件都已在索引中）
            if "nothing to commit" in result.stderr.lower():
                logger.debug(f"No new changes to commit")
                return True
            else:
                logger.warning(f"Failed to commit: {result.stderr}")
                return True  # 继续执行，不是致命错误

    except Exception as e:
        logger.warning(f"Failed to commit uncommitted changes: {e}")
        return True  # 继续执行，不是致命错误


def pull_latest_changes(repo_path: str) -> bool:
    """
    拉取远程仓库最新变更

    Args:
        repo_path: 仓库路径

    Returns:
        bool: 是否成功拉取（或无需拉取）
    """
    try:
        # 检查是否有远程仓库
        result = run_git_command(repo_path, ["remote", "-v"], check=False)
        if result.returncode != 0 or not result.stdout.strip():
            logger.debug(f"No remote repository configured for {repo_path}")
            return True

        logger.info(f"Pulling latest changes for {repo_path}")
        result = run_git_command(repo_path, ["pull", "--depth=1"], check=False)

        if result.returncode == 0:
            logger.info(f"Successfully pulled latest changes")
            return True
        else:
            logger.warning(f"Git pull failed (may be expected for local repos): {result.stderr}")
            return True  # 对于本地仓库，pull 失败不影响使用

    except Exception as e:
        logger.warning(f"Failed to pull latest changes: {e}")
        return True  # 继续执行，pull 失败不是致命错误


def detect_changes(repo_path: str, previous_hash: Optional[str] = None) -> GitChanges:
    """
    检测 git 仓库中的文件变更

    Args:
        repo_path: 仓库路径
        previous_hash: 上一个 commit hash（如果为 None，则自动检测）

    Returns:
        GitChanges: 变更信息
    """
    current_hash = get_current_commit(repo_path)

    if previous_hash is None:
        previous_hash = get_previous_commit(repo_path)

    is_initial = (previous_hash is None)
    changes: List[FileChange] = []

    if is_initial:
        # 首次提交，所有文件都是新增的
        logger.info("Initial commit - detecting all files as new")

        try:
            result = run_git_command(repo_path, ["ls-files"], check=True)
            all_files = result.stdout.strip().split('\n') if result.stdout.strip() else []

            for file_path in all_files:
                if file_path:  # 跳过空行
                    changes.append(FileChange(
                        change_type=FileChangeType.ADDED,
                        file_path=file_path
                    ))

            logger.info(f"Initial commit: detected {len(changes)} files")

        except Exception as e:
            logger.error(f"Failed to list files for initial commit: {e}")

    else:
        # 有历史记录，检测增量变更
        logger.info(f"Detecting changes between {previous_hash[:8]} and {current_hash[:8]}")

        try:
            # 获取变更文件列表
            result = run_git_command(
                repo_path,
                ["diff", "--name-status", f"{previous_hash}..{current_hash}"],
                check=True
            )

            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

            for line in lines:
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    logger.debug(f"Skipping invalid line (not enough parts): {line}")
                    continue

                status_code = parts[0]
                if not status_code:
                    logger.debug(f"Skipping invalid line (empty status code): {line}")
                    continue

                file_path = parts[1]

                # 解析 git 状态码
                # https://git-scm.com/docs/git-diff#Documentation/git-diff.txt---name-status
                change_type = None
                old_path = None

                if status_code == 'A':
                    change_type = FileChangeType.ADDED
                elif status_code == 'M':
                    change_type = FileChangeType.MODIFIED
                elif status_code == 'D':
                    change_type = FileChangeType.DELETED
                elif status_code.startswith('R'):
                    # 重命名：R100 或 Rxxx
                    change_type = FileChangeType.RENAMED
                    if len(parts) >= 3:
                        old_path = parts[1]
                        file_path = parts[2]
                elif status_code == 'T':
                    # 文件类型变更，视为修改
                    change_type = FileChangeType.MODIFIED

                if change_type:
                    changes.append(FileChange(
                        change_type=change_type,
                        file_path=file_path,
                        old_path=old_path
                    ))
                else:
                    logger.debug(f"Unknown status code '{status_code}' for file: {file_path}")

            logger.info(f"Detected {len(changes)} changes:")
            logger.info(f"  - Added: {sum(1 for c in changes if c.change_type == FileChangeType.ADDED)}")
            logger.info(f"  - Modified: {sum(1 for c in changes if c.change_type == FileChangeType.MODIFIED)}")
            logger.info(f"  - Deleted: {sum(1 for c in changes if c.change_type == FileChangeType.DELETED)}")
            logger.info(f"  - Renamed: {sum(1 for c in changes if c.change_type == FileChangeType.RENAMED)}")

        except Exception as e:
            logger.error(f"Failed to detect changes: {e}")
            # 降级为全量处理
            logger.warning("Falling back to full refresh due to error")
            is_initial = True

    return GitChanges(
        commit_hash=current_hash,
        previous_hash=previous_hash,
        changes=changes,
        is_initial=is_initial
    )


def get_file_at_commit(repo_path: str, file_path: str, commit_hash: str) -> Optional[str]:
    """
    获取指定 commit 中的文件内容

    Args:
        repo_path: 仓库路径
        file_path: 文件路径（相对于仓库根目录）
        commit_hash: commit hash

    Returns:
        Optional[str]: 文件内容，如果失败返回 None
    """
    try:
        result = run_git_command(
            repo_path,
            ["show", f"{commit_hash}:{file_path}"],
            check=False
        )

        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning(f"Failed to get file {file_path} at commit {commit_hash[:8]}")
            return None

    except Exception as e:
        logger.error(f"Error getting file {file_path} at commit {commit_hash[:8]}: {e}")
        return None


def get_current_file_content(repo_path: str, file_path: str) -> Optional[str]:
    """
    获取工作目录中文件的当前内容

    Args:
        repo_path: 仓库路径
        file_path: 文件路径（相对于仓库根目录）

    Returns:
        Optional[str]: 文件内容，如果失败返回 None
    """
    full_path = os.path.join(repo_path, file_path)

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(full_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


def save_commit_state(db_path: str, commit_hash: str) -> bool:
    """
    保存 commit hash 到数据库状态文件

    Args:
        db_path: 数据库文件路径
        commit_hash: commit hash

    Returns:
        bool: 是否成功保存
    """
    try:
        state_file = db_path.replace('.pkl', '_commit.txt')
        with open(state_file, 'w') as f:
            f.write(commit_hash)
        logger.debug(f"Saved commit state: {commit_hash[:8]} -> {state_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save commit state: {e}")
        return False


def load_commit_state(db_path: str) -> Optional[str]:
    """
    从数据库状态文件加载 commit hash

    Args:
        db_path: 数据库文件路径

    Returns:
        Optional[str]: commit hash，如果不存在返回 None
    """
    try:
        state_file = db_path.replace('.pkl', '_commit.txt')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                commit_hash = f.read().strip()
            logger.debug(f"Loaded commit state: {commit_hash[:8]} <- {state_file}")
            return commit_hash
        return None
    except Exception as e:
        logger.error(f"Failed to load commit state: {e}")
        return None
