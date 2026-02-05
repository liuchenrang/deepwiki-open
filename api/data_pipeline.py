import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import json
import tiktoken
import logging
import base64
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from api.ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from api.tools.embedder import get_embedder

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192

def count_tokens(text: str, embedder_type: str = None, is_ollama_embedder: bool = None) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama', 'bedrock').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        
        # Determine embedder type if not specified
        if embedder_type is None:
            from api.config import get_embedder_type
            embedder_type = get_embedder_type()

        # Choose encoding based on embedder type
        if embedder_type == 'ollama':
            # Ollama typically uses cl100k_base encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        elif embedder_type == 'google':
            # Google uses similar tokenization to GPT models for rough estimation
            encoding = tiktoken.get_encoding("cl100k_base")
        elif embedder_type == 'bedrock':
            # Bedrock embedding models vary; use a common GPT-like encoding for rough estimation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:  # OpenAI or default
            # Use OpenAI embedding model encoding
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, repo_type: str = None, access_token: str = None) -> str:
    """
    Downloads a Git repository (GitHub, GitLab, or Bitbucket) to a specified local path.

    Args:
        repo_type(str): Type of repository
        repo_url (str): The URL of the Git repository to clone.
        local_path (str): The local directory where the repository will be cloned.
        access_token (str, optional): Access token for private repositories.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check if repository already exists
        if os.path.exists(local_path) and os.listdir(local_path):
            # Directory exists and is not empty
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Prepare the clone URL with access token if provided
        clone_url = repo_url
        if access_token:
            parsed = urlparse(repo_url)
            # URL-encode the token to handle special characters
            encoded_token = quote(access_token, safe='')
            # Determine the repository type and format the URL accordingly
            if repo_type == "github":
                # Format: https://{token}@{domain}/owner/repo.git
                # Works for both github.com and enterprise GitHub domains
                clone_url = urlunparse((parsed.scheme, f"{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif repo_type == "gitlab":
                # Format: https://oauth2:{token}@gitlab.com/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"oauth2:{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif repo_type == "bitbucket":
                # Format: https://x-token-auth:{token}@bitbucket.org/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"x-token-auth:{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))

            logger.info("Using access token for authentication")

        # Clone the repository
        logger.info(f"Cloning repository from {repo_url} to {local_path}")
        # We use repo_url in the log to avoid exposing the token in logs
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", clone_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("Repository cloned successfully")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        # Sanitize error message to remove any tokens (both raw and URL-encoded)
        if access_token:
            # Remove raw token
            error_msg = error_msg.replace(access_token, "***TOKEN***")
            # Also remove URL-encoded token to prevent leaking encoded version
            encoded_token = quote(access_token, safe='')
            error_msg = error_msg.replace(encoded_token, "***TOKEN***")
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

# Alias for backward compatibility
download_github_repo = download_repo

def read_all_documents(path: str, embedder_type: str = None, is_ollama_embedder: bool = None, 
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.
        excluded_dirs (List[str], optional): List of directories to exclude from processing.
            Overrides the default configuration if provided.
        excluded_files (List[str], optional): List of file patterns to exclude from processing.
            Overrides the default configuration if provided.
        included_dirs (List[str], optional): List of directories to include exclusively.
            When provided, only files in these directories will be processed.
        included_files (List[str], optional): List of file patterns to include exclusively.
            When provided, only files matching these patterns will be processed.

    Returns:
        list: A list of Document objects with metadata.
    """
    # Handle backward compatibility
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # Determine filtering mode: inclusion or exclusion
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

    if use_inclusion_mode:
        # Inclusion mode: only process specified directories and files
        final_included_dirs = set(included_dirs) if included_dirs else set()
        final_included_files = set(included_files) if included_files else set()

        logger.info(f"Using inclusion mode")
        logger.info(f"Included directories: {list(final_included_dirs)}")
        logger.info(f"Included files: {list(final_included_files)}")

        # Convert to lists for processing
        included_dirs = list(final_included_dirs)
        included_files = list(final_included_files)
        excluded_dirs = []
        excluded_files = []
    else:
        # Exclusion mode: use default exclusions plus any additional ones
        final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files = set(DEFAULT_EXCLUDED_FILES)

        # Add any additional excluded directories from config
        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        # Add any additional excluded files from config
        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        # Add any explicitly provided excluded directories and files
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        # Convert back to lists for compatibility
        excluded_dirs = list(final_excluded_dirs)
        excluded_files = list(final_excluded_files)
        included_dirs = []
        included_files = []

        logger.info(f"Using exclusion mode")
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

    logger.info(f"Reading documents from {path}")

    def should_process_file(file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.

        Args:
            file_path (str): The file path to check
            use_inclusion (bool): Whether to use inclusion mode
            included_dirs (List[str]): List of directories to include
            included_files (List[str]): List of files to include
            excluded_dirs (List[str]): List of directories to exclude
            excluded_files (List[str]): List of files to exclude

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        file_path_parts = os.path.normpath(file_path).split(os.sep)
        file_name = os.path.basename(file_path)

        if use_inclusion:
            # Inclusion mode: file must be in included directories or match included files
            is_included = False

            # Check if file is in an included directory
            if included_dirs:
                for included in included_dirs:
                    clean_included = included.strip("./").rstrip("/")
                    if clean_included in file_path_parts:
                        is_included = True
                        break

            # Check if file matches included file patterns
            if not is_included and included_files:
                for included_file in included_files:
                    if file_name == included_file or file_name.endswith(included_file):
                        is_included = True
                        break

            # If no inclusion rules are specified for a category, allow all files from that category
            if not included_dirs and not included_files:
                is_included = True
            elif not included_dirs and included_files:
                # Only file patterns specified, allow all directories
                pass  # is_included is already set based on file patterns
            elif included_dirs and not included_files:
                # Only directory patterns specified, allow all files in included directories
                pass  # is_included is already set based on directory patterns

            return is_included
        else:
            # Exclusion mode: file must not be in excluded directories or match excluded files
            is_excluded = False

            # Check if file is in an excluded directory
            for excluded in excluded_dirs:
                clean_excluded = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            # Check if file matches excluded file patterns
            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                # Get relative path first
                relative_path = os.path.relpath(file_path, path)

                # Try UTF-8 first, fallback to latin-1 for binary/non-UTF8 files
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Fallback to latin-1 which can read any byte sequence
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                    logger.warning(f"File {relative_path} is not UTF-8 encoded, read as latin-1")

                # Determine if this is an implementation file
                is_implementation = (
                    not relative_path.startswith("test_")
                    and not relative_path.startswith("app_")
                    and "test" not in relative_path.lower()
                )

                # Check token count
                token_count = count_tokens(content, embedder_type)
                if token_count > MAX_EMBEDDING_TOKENS * 10:
                    logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                    continue

                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:],
                        "is_code": True,
                        "is_implementation": is_implementation,
                        "title": relative_path,
                        "token_count": token_count,
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                # Get relative path first
                relative_path = os.path.relpath(file_path, path)

                # Try UTF-8 first, fallback to latin-1 for binary/non-UTF8 files
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Fallback to latin-1 which can read any byte sequence
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                    logger.warning(f"File {relative_path} is not UTF-8 encoded, read as latin-1")

                    # Check token count
                    token_count = count_tokens(content, embedder_type)
                    if token_count > MAX_EMBEDDING_TOKENS:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def prepare_data_pipeline(embedder_type: str = None, is_ollama_embedder: bool = None, repo_name: str = None):
    """
    Creates and returns the data transformation pipeline.

    Args:
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.
        repo_name (str, optional): Repository name for cache isolation. If provided,
                                  will be used to create unique cache files per repository.

    Returns:
        adal.Sequential: The data transformation pipeline
    """
    from api.config import get_embedder_config, get_embedder_type

    # Handle backward compatibility
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None

    # Determine embedder type if not specified
    if embedder_type is None:
        embedder_type = get_embedder_type()

    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()

    embedder = get_embedder(embedder_type=embedder_type)

    # Choose appropriate processor based on embedder type
    if embedder_type == 'ollama':
        # Use Ollama document processor for single-document processing
        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
    elif embedder_type == 'dashscope':
        # Use DashScope-specific batch processor with proper batch size limits
        from api.dashscope_client import DashScopeToEmbeddings
        batch_size = embedder_config.get("batch_size", 10)  # DashScope API 限制为 10
        # 使用 repo_name 作为缓存文件名的一部分，确保每个仓库有独立的缓存
        cache_suffix = repo_name if repo_name else "embeddings"
        embedder_transformer = DashScopeToEmbeddings(
            embedder=embedder,
            batch_size=batch_size,
            embedding_cache_file_name=f"{embedder_type}_{cache_suffix}"
        )
    else:
        # Use batch processing for OpenAI and Google embedders
        batch_size = embedder_config.get("batch_size", 500)
        embedder_transformer = ToEmbeddings(
            embedder=embedder, batch_size=batch_size
        )

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer

def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, embedder_type: str = None, is_ollama_embedder: bool = None,
    repo_name: str = None
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.
        repo_name (str, optional): Repository name for cache isolation.
    """
    # Get the data transformer
    data_transformer = prepare_data_pipeline(embedder_type, is_ollama_embedder, repo_name)

    # Save the documents to a local database
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitHub repository using the GitHub API.
    Supports both public GitHub (github.com) and GitHub Enterprise (custom domains).
    
    Args:
        repo_url (str): The URL of the GitHub repository 
                       (e.g., "https://github.com/username/repo" or "https://github.company.com/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): GitHub personal access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not a valid GitHub URL
    """
    try:
        # Parse the repository URL to support both github.com and enterprise GitHub
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        # Check if it's a GitHub-like URL structure
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # Determine the API base URL
        if parsed_url.netloc == "github.com":
            # Public GitHub
            api_base = "https://api.github.com"
        else:
            # GitHub Enterprise - API is typically at https://domain/api/v3/
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"
        
        # Use GitHub API to get file content
        # The API endpoint for getting file content is: /repos/{owner}/{repo}/contents/{path}
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # Fetch file content from GitHub API
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        logger.info(f"Fetching file content from GitHub API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response from GitHub API")

        # Check if we got an error response
        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API error: {content_data['message']}")

        # GitHub API returns file content as base64 encoded string
        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                # The content might be split into lines, so join them first
                content_base64 = content_data["content"].replace("\n", "")
                content = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unexpected encoding: {content_data['encoding']}")
        else:
            raise ValueError("File content not found in GitHub API response")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository (cloud or self-hosted).

    Args:
        repo_url (str): The GitLab repo URL (e.g., "https://gitlab.com/username/repo" or "http://localhost/group/project")
        file_path (str): File path within the repository (e.g., "src/main.py")
        access_token (str, optional): GitLab personal access token

    Returns:
        str: File content

    Raises:
        ValueError: If anything fails
    """
    try:
        # Parse and validate the URL
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format — expected something like https://gitlab.domain.com/group/project")

        # Build project path and encode for API
        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')

        # Encode file path
        encoded_file_path = quote(file_path, safe='')

        # Try to get the default branch from the project info
        default_branch = None
        try:
            project_info_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}"
            project_headers = {}
            if access_token:
                project_headers["PRIVATE-TOKEN"] = access_token
            
            project_response = requests.get(project_info_url, headers=project_headers)
            if project_response.status_code == 200:
                project_data = project_response.json()
                default_branch = project_data.get('default_branch', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning(f"Could not fetch project info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching project info: {e}, using 'main' as default branch")
            default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        # Fetch file content from GitLab API
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"Fetching file content from GitLab API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

        # Check for GitLab error response (JSON instead of raw file)
        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API error: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Bitbucket repository using the Bitbucket API.

    Args:
        repo_url (str): The URL of the Bitbucket repository (e.g., "https://bitbucket.org/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): Bitbucket personal access token for private repositories

    Returns:
        str: The content of the file as a string
    """
    try:
        # Extract owner and repo name from Bitbucket URL
        if not (repo_url.startswith("https://bitbucket.org/") or repo_url.startswith("http://bitbucket.org/")):
            raise ValueError("Not a valid Bitbucket repository URL")

        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            raise ValueError("Invalid Bitbucket URL format")

        owner = parts[-2]
        repo = parts[-1].replace(".git", "")

        # Try to get the default branch from the repository info
        default_branch = None
        try:
            repo_info_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
            repo_headers = {}
            if access_token:
                repo_headers["Authorization"] = f"Bearer {access_token}"
            
            repo_response = requests.get(repo_info_url, headers=repo_headers)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                default_branch = repo_data.get('mainbranch', {}).get('name', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning(f"Could not fetch repository info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching repository info: {e}, using 'main' as default branch")
            default_branch = 'main'

        # Use Bitbucket API to get file content
        # The API endpoint for getting file content is: /2.0/repositories/{owner}/{repo}/src/{branch}/{path}
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/{default_branch}/{file_path}"

        # Fetch file content from Bitbucket API
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        logger.info(f"Fetching file content from Bitbucket API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                content = response.text
            elif response.status_code == 404:
                raise ValueError("File not found on Bitbucket. Please check the file path and repository.")
            elif response.status_code == 401:
                raise ValueError("Unauthorized access to Bitbucket. Please check your access token.")
            elif response.status_code == 403:
                raise ValueError("Forbidden access to Bitbucket. You might not have permission to access this file.")
            elif response.status_code == 500:
                raise ValueError("Internal server error on Bitbucket. Please try again later.")
            else:
                response.raise_for_status()
                content = response.text
            return content
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def get_file_content(repo_url: str, file_path: str, repo_type: str = None, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Git repository (GitHub or GitLab).

    Args:
        repo_type (str): Type of repository
        repo_url (str): The URL of the repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): Access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not valid
    """
    if repo_type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif repo_type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif repo_type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository type. Only GitHub, GitLab, and Bitbucket are supported.")

class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    Supports both traditional LocalDB (faiss) and vector database backends (pgvector).
    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None
        self.vector_db = None  # Vector database backend (pgvector or faiss)
        self.use_vector_db = False  # Whether to use vector database backend

    def prepare_database(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None,
                         embedder_type: str = None, is_ollama_embedder: bool = None,
                         excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                         included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Create a new database from the repository.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
            embedder_type (str, optional): Embedder type to use ('openai', 'google', 'ollama').
                                         If None, will be determined from configuration.
            is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                               If None, will be determined from configuration.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Document]: List of Document objects
        """
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None

        logger.info("=" * 80)
        logger.info("🔍 [DEBUG] prepare_database() 开始")
        logger.info("=" * 80)

        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token)

        result = self.prepare_db_index(embedder_type=embedder_type, excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files)

        logger.info("=" * 80)
        logger.info("🔍 [DEBUG] prepare_database() 即将返回")
        logger.info(f"   → 返回文档数: {len(result) if result else 0}")
        logger.info(f"   → self.use_vector_db: {self.use_vector_db}")
        logger.info(f"   → self.vector_db: {self.vector_db}")
        logger.info("=" * 80)

        return result

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None
        self.vector_db = None
        self.use_vector_db = False

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        # Extract owner and repo name to create unique identifier
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            # Bitbucket URL format: https://bitbucket.org/owner/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name

    def _init_vector_db_backend(self, repo_url_or_path: str, repo_type: str = None):
        """
        Initialize vector database backend if VECTOR_DB_BACKEND is configured.

        Args:
            repo_url_or_path: Repository URL or local path
            repo_type: Repository type (github, gitlab, bitbucket, local)

        Raises:
            ImportError: 如果 pgvector 依赖缺失
            Exception: 如果初始化失败
        """
        logger.info("=" * 80)
        logger.info("🔍 [VECTOR DB] 开始初始化向量数据库后端")
        logger.info(f"   仓库路径: {repo_url_or_path}")
        logger.info(f"   仓库类型: {repo_type}")

        # 检查环境变量
        vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "faiss")
        logger.info(f"   环境变量 VECTOR_DB_BACKEND: {vector_db_backend}")

        # 如果配置的是 pgvector，必须成功初始化，否则报错
        if vector_db_backend == "pgvector":
            logger.info("   → 配置要求使用 pgvector 后端")
            logger.info("   → 正在尝试创建向量数据库实例...")

            try:
                from api.vector_db import get_vector_db

                logger.info("   → 正在创建向量数据库实例...")
                vector_db = get_vector_db()
                logger.info(f"   ✅ 向量数据库实例创建成功: {type(vector_db).__name__}")
                logger.info(f"   → vector_db.config 类型: {type(vector_db.config)}")
                logger.info(f"   → vector_db.config 内容: {vector_db.config}")
                logger.info(f"   → config.get('backend'): {vector_db.config.get('backend', 'NOT_FOUND')}")

                # Set repository context for pgvector
                if hasattr(vector_db, 'use_repository'):
                    # Extract owner and repo from URL
                    owner = "local"
                    repo = "unknown"

                    if repo_type in ["github", "gitlab", "bitbucket"]:
                        url_parts = repo_url_or_path.rstrip('/').split('/')
                        if len(url_parts) >= 5:
                            owner = url_parts[-2]
                            repo = url_parts[-1].replace(".git", "")
                    else:
                        # Local path
                        repo = os.path.basename(repo_url_or_path)

                    logger.info(f"   → 正在设置仓库上下文: {owner}/{repo}")
                    vector_db.use_repository(owner=owner, repo=repo, repo_url=repo_url_or_path)
                    logger.info(f"   ✅ 仓库上下文设置成功")

                    # 🔍 调试日志：确认 repository_id 已设置
                    if hasattr(vector_db, 'repository_id'):
                        logger.info(f"   → vector_db.repository_id: {vector_db.repository_id}")

                self.vector_db = vector_db
                self.use_vector_db = True

                # 🔍 调试日志：确认状态已设置
                logger.info(f"   → self.vector_db: {self.vector_db}")
                logger.info(f"   → self.use_vector_db: {self.use_vector_db}")

                logger.info(f"🎉 [VECTOR DB] 向量数据库后端初始化完成: {type(vector_db).__name__}")
                logger.info(f"   → 后端类型: {type(vector_db).__name__}")
                logger.info(f"   → 向量维度: {vector_db.embedding_dimension}")
                logger.info("=" * 80)

            except ImportError as e:
                # pgvector 依赖缺失，明确报错
                error_msg = f"""
❌ pgvector 后端初始化失败：缺少必需的依赖包

错误详情: {str(e)}

💡 解决方案：

1. 重新构建 Docker 镜像（推荐）：
   docker-compose down
   docker-compose build deepwiki
   docker-compose up -d

2. 或在容器中手动安装：
   docker-compose exec deepwiki pip install psycopg[binary] pgvector

3. 或在开发环境中安装：
   pip install psycopg[binary] pgvector

📝 配置文件: api/pyproject.toml
   确保 poetry.lock 已更新
"""
                logger.error("=" * 80)
                logger.error(error_msg)
                logger.error("=" * 80)
                raise ImportError(error_msg) from e

            except Exception as e:
                # 其他初始化错误，也明确报错
                error_msg = f"""
❌ pgvector 后端初始化失败

配置要求: VECTOR_DB_BACKEND=pgvector
仓库类型: {repo_type}
仓库路径: {repo_url_or_path}

错误类型: {type(e).__name__}
错误详情: {str(e)}

💡 可能的原因：
1. PostgreSQL 连接失败（检查 DATABASE_URL 配置）
2. pgvector 扩展未安装（需要在 PostgreSQL 中运行 CREATE EXTENSION vector）
3. 数据库权限不足

📝 故障排查：
   docker-compose exec deepwiki python -c "
   import psycopg
   from pgvector.psycopg import register_vector
   conn = psycopg.connect(os.getenv('DATABASE_URL'))
   register_vector(conn)
   print('✅ 连接成功')
   "
"""
                logger.error("=" * 80)
                logger.error(error_msg)
                logger.error("=" * 80)
                raise RuntimeError(f"pgvector backend initialization failed: {str(e)}") from e
        else:
            # 配置的是 faiss 或未设置，不使用向量数据库
            logger.info("=" * 80)
            logger.info(f"ℹ️  [VECTOR DB] 配置使用传统 FAISS 后端")
            logger.info(f"   → VECTOR_DB_BACKEND: {vector_db_backend}")
            logger.info(f"   → 数据将保存在本地文件系统 (~/.adalflow/databases/)")
            logger.info("=" * 80)

            self.vector_db = None
            self.use_vector_db = False

    def _create_repo(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.adalflow/repos/{owner}_{repo_name} (for url, local path will be the same)
        ~/.adalflow/databases/{owner}_{repo_name}.pkl

        Git 增量更新支持：
        - 对于远程仓库：clone 后使用 git pull 获取最新变更
        - 对于本地路径：如果不是 git 仓库，自动初始化为 git 仓库

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
        """
        logger.info(f"Preparing repo storage for {repo_url_or_path}...")
        from api.git_utils import (
            is_git_repository, initialize_git_repo, pull_latest_changes, commit_uncommitted_changes
        )

        try:
            # Strip whitespace to handle URLs with leading/trailing spaces
            repo_url_or_path = repo_url_or_path.strip()

            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # Extract the repository name from the URL
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"Extracted repo name: {repo_name}")

                save_repo_dir = os.path.join(root_path, "repos", repo_name)

                # Check if the repository directory already exists and is not empty
                if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                    # Only download if the repository doesn't exist or is empty
                    download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
                else:
                    logger.info(f"Repository already exists at {save_repo_dir}. Pulling latest changes...")
                    # 拉取最新变更
                    pull_latest_changes(save_repo_dir)
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

                # 对于本地路径，如果不是 git 仓库则初始化
                if not is_git_repository(save_repo_dir):
                    logger.info(f"Local path {save_repo_dir} is not a git repository. Initializing...")
                    initialize_git_repo(save_repo_dir)
                else:
                    logger.info(f"Local path {save_repo_dir} is already a git repository")

                    # ⚠️ 关键修复：自动提交未提交的更改
                    logger.info(f"Checking for uncommitted changes...")
                    commit_uncommitted_changes(save_repo_dir)

                    # 尝试拉取最新变更（如果有远程仓库）
                    pull_latest_changes(save_repo_dir)

            save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.info(f"Repo paths: {self.repo_paths}")

            # Initialize vector database backend if configured
            self._init_vector_db_backend(repo_url_or_path, repo_type)

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(self, embedder_type: str = None, is_ollama_embedder: bool = None,
                        excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Prepare the indexed database for the repository with git-based incremental updates.

        Git 增量更新逻辑：
        1. 检查当前 commit hash 和上次保存的 commit hash
        2. 如果相同，直接加载现有数据库
        3. 如果不同，检测变更并增量处理：
           - 删除的文件：从向量数据库中删除
           - 新增/修改的文件：重新生成向量并添加

        Args:
            embedder_type (str, optional): Embedder type to use ('openai', 'google', 'ollama').
                                         If None, will be determined from configuration.
            is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                               If None, will be determined from configuration.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Document]: List of Document objects
        """
        from api.git_utils import (
            get_current_commit, load_commit_state, save_commit_state,
            detect_changes, FileChangeType
        )

        def _embedding_vector_length(doc: Document) -> int:
            vector = getattr(doc, "vector", None)
            if vector is None:
                return 0
            try:
                if hasattr(vector, "shape"):
                    if len(vector.shape) == 0:
                        return 0
                    return int(vector.shape[-1])
                if hasattr(vector, "__len__"):
                    return int(len(vector))
            except Exception:
                return 0
            return 0

        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None

        # ========== Git 增量检测 ==========
        logger.info("=" * 80)
        logger.info("🔍 [GIT INCREMENTAL] 开始检测代码变更")

        current_commit = get_current_commit(self.repo_paths["save_repo_dir"])

        # 🔧 关键修复：使用 pgvector 时从数据库读取 commit 状态
        if self.use_vector_db and self.vector_db and hasattr(self.vector_db, 'get_current_commit'):
            saved_commit = self.vector_db.get_current_commit()
            logger.info(f"   → 从 pgvector 数据库读取 commit 状态")
        else:
            saved_commit = load_commit_state(self.repo_paths["save_db_file"])

        logger.info(f"   → 当前 commit: {current_commit[:8] if current_commit else 'N/A'}")
        logger.info(f"   → 保存的 commit: {saved_commit[:8] if saved_commit else 'N/A'}")

        # 如果 commit 相同且数据库存在，直接加载
        if current_commit and current_commit == saved_commit and os.path.exists(self.repo_paths["save_db_file"]):
            logger.info("✅ [GIT INCREMENTAL] 代码未变更，直接加载现有数据库")
            logger.info("=" * 80)

            self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
            transformed_docs = self.db.get_transformed_data(key="split_and_embed")

            if transformed_docs:
                logger.info(f"✅ 已加载 {len(transformed_docs)} 个文档（无需重新生成）")

                # 🔧 关键修复：即使 commit 相同，也检查并同步向量数据库
                if self.use_vector_db and self.vector_db:
                    try:
                        logger.info("🧹 [SYNC] 检查向量数据库状态")

                        with self.vector_db._get_connection() as conn:
                            with conn.cursor() as cur:
                                # 检查 pgvector 中是否有该仓库的向量
                                cur.execute("""
                                    SELECT
                                        COUNT(*) as total,
                                        COUNT(DISTINCT file_path) as unique_files
                                    FROM document_chunks
                                    WHERE repository_id = %s AND deleted = FALSE
                                """, (self.vector_db.repository_id,))

                                pgvector_total, pgvector_files = cur.fetchone()

                                logger.info(f"   → pgvector: {pgvector_total} 个向量, {pgvector_files} 个文件")
                                logger.info(f"   → LocalDB: {len(transformed_docs)} 个文档")

                                # 如果 pgvector 是空的或文件数不匹配，需要同步
                                if pgvector_total == 0 or pgvector_files == 0:
                                    logger.info(f"   → pgvector 为空，开始同步 LocalDB 文档...")

                                    # 清理重复向量（如果有）
                                    cur.execute("""
                                        SELECT file_path, COUNT(*) as count
                                        FROM document_chunks
                                        WHERE repository_id = %s AND deleted = FALSE
                                        GROUP BY file_path
                                        HAVING COUNT(*) > 1
                                    """, (self.vector_db.repository_id,))

                                    duplicates = cur.fetchall()

                                    if duplicates:
                                        logger.info(f"   → 发现 {len(duplicates)} 个文件有重复向量，先清理")
                                        for file_path, count in duplicates:
                                            cur.execute("""
                                                DELETE FROM document_chunks
                                                WHERE repository_id = %s AND file_path = %s
                                            """, (self.vector_db.repository_id, file_path))
                                        conn.commit()

                                    # 同步所有文档到 pgvector
                                    logger.info(f"   → 正在同步 {len(transformed_docs)} 个文档到 pgvector")
                                    doc_ids = self.vector_db.add_documents(transformed_docs)
                                    logger.info(f"   ✅ 成功同步 {len(doc_ids)} 个文档到 pgvector")

                                    # 更新 commit 状态
                                    if current_commit:
                                        self.vector_db.update_current_commit(current_commit)
                                        logger.info(f"   ✅ 更新 commit 状态: {current_commit[:8]}")
                                else:
                                    # pgvector 有数据，只检查和清理重复
                                    logger.info(f"   → pgvector 有数据，检查重复向量")

                                    cur.execute("""
                                        SELECT file_path, COUNT(*) as count
                                        FROM document_chunks
                                        WHERE repository_id = %s AND deleted = FALSE
                                        GROUP BY file_path
                                        HAVING COUNT(*) > 1
                                    """, (self.vector_db.repository_id,))

                                    duplicates = cur.fetchall()

                                    if duplicates:
                                        logger.info(f"   → 发现 {len(duplicates)} 个文件有重复向量，开始清理")
                                        cleaned_count = 0
                                        for file_path, count in duplicates:
                                            cur.execute("""
                                                DELETE FROM document_chunks
                                                WHERE id IN (
                                                    SELECT id FROM (
                                                        SELECT id,
                                                               ROW_NUMBER() OVER (
                                                                   PARTITION BY repository_id, file_path, chunk_index
                                                                   ORDER BY created_at DESC
                                                               ) as rn
                                                        FROM document_chunks
                                                        WHERE repository_id = %s
                                                          AND file_path = %s
                                                          AND deleted = FALSE
                                                    ) sub WHERE rn > 1
                                                )
                                            """, (self.vector_db.repository_id, file_path))

                                            deleted = cur.rowcount
                                            cleaned_count += deleted
                                            logger.info(f"      - 清理 {file_path}: 删除 {deleted} 个旧版本")

                                        conn.commit()
                                        logger.info(f"   ✅ 清理完成: 删除了 {cleaned_count} 个重复向量")
                                    else:
                                        logger.info(f"   ✅ 没有发现重复向量")

                    except Exception as e:
                        logger.warning(f"   ⚠️ 向量数据库同步失败: {e}")
                        import traceback
                        logger.warning(f"   详细错误: {traceback.format_exc()}")

                return transformed_docs
            else:
                logger.warning("⚠️ 数据库存在但无文档，将重新处理")

        # ========== 判断是新建仓库还是已存在仓库 ==========
        is_new_repository = False

        if self.use_vector_db and self.vector_db and hasattr(self.vector_db, 'count_documents'):
            try:
                # 检查数据库中该仓库的文档数量
                existing_doc_count = self.vector_db.count_documents()

                logger.info("=" * 80)
                logger.info("🔍 [REPOSITORY CHECK] 检查仓库状态")
                logger.info(f"   → repository_id: {self.vector_db.repository_id}")
                logger.info(f"   → 数据库中现有文档数: {existing_doc_count}")

                if existing_doc_count == 0:
                    is_new_repository = True
                    logger.info(f"   ✅ 判断为: 新建仓库（数据库无文档）")
                    logger.info(f"   → 将执行全量处理")
                else:
                    is_new_repository = False
                    logger.info(f"   ✅ 判断为: 已存在仓库（数据库有 {existing_doc_count} 个文档）")
                    logger.info(f"   → 将执行增量更新")

                logger.info("=" * 80)

            except Exception as e:
                logger.warning(f"   ⚠️ 无法检查仓库状态: {e}")
                # 如果无法检查，默认作为已存在仓库处理（使用增量更新）
                is_new_repository = False
        else:
            # 不使用向量数据库时，使用 commit 判断
            is_new_repository = (saved_commit is None or saved_commit == 'N/A')

        # ========== 检测 git 变更 ==========
        git_changes = detect_changes(self.repo_paths["save_repo_dir"], previous_hash=saved_commit)

        if git_changes.is_initial:
            logger.info("🆕 [GIT INCREMENTAL] Git 检测为首次处理")
        elif is_new_repository:
            logger.info("🆕 [REPOSITORY] 数据库判断为新建仓库")
        else:
            logger.info(f"📝 [GIT INCREMENTAL] 检测到 {len(git_changes.changes)} 个文件变更")

        # ========== 准备文档列表 ==========
        transformed_docs = None

        # 🔧 关键判断：新建仓库或 Git 首次处理 → 全量处理
        # 检查本地数据库是否存在（已存在仓库的增量更新）
        if (is_new_repository or git_changes.is_initial):
            # 新建仓库：直接跳到全量处理
            logger.info("⏭️  跳过增量更新逻辑，直接进入全量处理")
            transformed_docs = None
        elif os.path.exists(self.repo_paths["save_db_file"]):
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                existing_docs = self.db.get_transformed_data(key="split_and_embed")

                if existing_docs:
                    logger.info(f"📦 加载现有数据库: {len(existing_docs)} 个文档")

                    # 🔍 调试日志：显示现有数据库中的文件
                    logger.info(f"   📄 现有数据库中的文件:")
                    for doc in existing_docs:
                        file_path = doc.meta_data.get('file_path', 'unknown')
                        logger.info(f"      - {file_path}")

                    # 创建现有文档的 file_path 集合
                    existing_file_paths = {
                        doc.meta_data.get('file_path') for doc in existing_docs
                        if doc.meta_data and 'file_path' in doc.meta_data
                    }

                    # 找出需要重新处理的文件（新增或修改）
                    files_to_reprocess = set()
                    for change in git_changes.changes:
                        if change.change_type in [FileChangeType.ADDED, FileChangeType.MODIFIED]:
                            files_to_reprocess.add(change.file_path)

                    # 移除已删除和重命名的文件
                    files_to_remove = set()
                    for change in git_changes.changes:
                        if change.change_type == FileChangeType.DELETED:
                            files_to_remove.add(change.file_path)
                        elif change.change_type == FileChangeType.RENAMED and change.old_path:
                            files_to_remove.add(change.old_path)

                    # 过滤掉需要移除的文档
                    if files_to_remove:
                        filtered_docs = [
                            doc for doc in existing_docs
                            if doc.meta_data.get('file_path') not in files_to_remove
                        ]
                        logger.info(f"   → 过滤掉 {len(existing_docs) - len(filtered_docs)} 个已删除文件的文档")
                        existing_docs = filtered_docs

                    # 如果有文件需要重新处理
                    if files_to_reprocess:
                        logger.info(f"🔄 重新处理 {len(files_to_reprocess)} 个变更的文件")
                        logger.info(f"   → 文件列表: {list(files_to_reprocess)[:10]}...")

                        # 重新读取这些文件并生成向量
                        new_documents = read_all_documents(
                            self.repo_paths["save_repo_dir"],
                            embedder_type=embedder_type,
                            excluded_dirs=excluded_dirs,
                            excluded_files=excluded_files,
                            included_dirs=included_dirs,
                            included_files=included_files
                        )

                        # 🔍 调试日志：显示所有读取到的文件
                        logger.info(f"   📄 读取到 {len(new_documents)} 个文件:")
                        for doc in new_documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            is_changed = file_path in files_to_reprocess
                            status = "✅ 变更" if is_changed else "⏩ 未变更"
                            logger.info(f"      {status} - {file_path}")

                        # 只保留需要重新处理的文件
                        reprocess_documents = [
                            doc for doc in new_documents
                            if doc.meta_data.get('file_path') in files_to_reprocess
                        ]

                        if reprocess_documents:
                            logger.info(f"   → 找到 {len(reprocess_documents)} 个需要重新处理的文档")

                            # 生成向量
                            repo_name = os.path.basename(self.repo_paths["save_db_file"]).replace('.pkl', '')
                            data_transformer = prepare_data_pipeline(embedder_type, is_ollama_embedder, repo_name)
                            reprocessed_docs = data_transformer(reprocess_documents)

                            # 🔧 关键修复：合并文档时移除旧版本，避免重复
                            # 创建文件路径到新文档的映射
                            reprocessed_file_paths = {
                                doc.meta_data.get('file_path') for doc in reprocessed_docs
                                if doc.meta_data and 'file_path' in doc.meta_data
                            }

                            # 只保留未重新处理的旧文档 + 新处理的文档
                            filtered_existing_docs = [
                                doc for doc in existing_docs
                                if doc.meta_data.get('file_path') not in reprocessed_file_paths
                            ]

                            # 合并文档
                            transformed_docs = filtered_existing_docs + reprocessed_docs
                            logger.info(f"   → 合并后文档总数: {len(transformed_docs)} (保留旧文档 {len(filtered_existing_docs)} + 新文档 {len(reprocessed_docs)})")
                        else:
                            transformed_docs = existing_docs
                    else:
                        # 没有文件需要重新处理，直接使用现有文档
                        transformed_docs = existing_docs
                        logger.info(f"✅ 无需重新处理，使用现有 {len(transformed_docs)} 个文档")

            except Exception as e:
                logger.error(f"❌ 加载现有数据库失败: {e}")
                logger.info("   → 将执行全量重建")
                transformed_docs = None

        # ========== 全量处理（首次或失败时） ==========
        if transformed_docs is None:
            logger.info("🔄 [FULL BUILD] 执行全量文档处理")

            documents = read_all_documents(
                self.repo_paths["save_repo_dir"],
                embedder_type=embedder_type,
                excluded_dirs=excluded_dirs,
                excluded_files=excluded_files,
                included_dirs=included_dirs,
                included_files=included_files
            )

            logger.info(f"   → 读取到 {len(documents)} 个文档")

            # 🔍 调试日志：显示所有文件
            logger.info(f"   📄 文件列表:")
            for doc in documents:
                file_path = doc.meta_data.get('file_path', 'unknown')
                file_type = doc.meta_data.get('type', 'unknown')
                is_code = doc.meta_data.get('is_code', False)
                code_str = "代码" if is_code else "文档"
                logger.info(f"      - [{code_str}] {file_path} (type: {file_type})")

            # Extract repo name from db file path for cache isolation
            repo_name = os.path.basename(self.repo_paths["save_db_file"]).replace('.pkl', '')
            self.db = transform_documents_and_save_to_db(
                documents, self.repo_paths["save_db_file"], embedder_type=embedder_type, repo_name=repo_name
            )

            logger.info(f"   → 生成向量并保存到数据库")
            transformed_docs = self.db.get_transformed_data(key="split_and_embed")
            logger.info(f"✅ 全量处理完成: {len(transformed_docs)} 个文档")

        # ========== 保存当前 commit 状态 ==========
        if current_commit:
            # 🔧 关键修复：使用 pgvector 时保存到数据库
            if self.use_vector_db and self.vector_db and hasattr(self.vector_db, 'update_current_commit'):
                self.vector_db.update_current_commit(current_commit)
                logger.info(f"💾 [PGVECTOR] 保存 commit 状态到数据库: {current_commit[:8]}")
            else:
                save_commit_state(self.repo_paths["save_db_file"], current_commit)
                logger.debug(f"💾 保存 commit 状态到文件: {current_commit[:8]}")

        # ========== 向量数据库增量更新 ==========
        # 🔍 调试日志：检查向量保存条件
        logger.info("=" * 80)
        logger.info("🔍 [DEBUG] 检查向量数据库保存条件")
        logger.info(f"   → self.use_vector_db: {self.use_vector_db}")
        logger.info(f"   → self.vector_db: {self.vector_db}")
        logger.info(f"   → vector_db 类型: {type(self.vector_db).__name__ if self.vector_db else 'None'}")
        logger.info(f"   → transformed_docs 数量: {len(transformed_docs) if transformed_docs else 0}")
        logger.info(f"   → transformed_docs 是否为 None: {transformed_docs is None}")

        # 检查第一个文档的向量状态
        if transformed_docs and len(transformed_docs) > 0:
            first_doc = transformed_docs[0]
            logger.info(f"   → 第一个文档有向量属性: {hasattr(first_doc, 'vector')}")
            if hasattr(first_doc, 'vector'):
                logger.info(f"   → 第一个文档向量值: {first_doc.vector}")
                logger.info(f"   → 第一个文档向量类型: {type(first_doc.vector)}")
                if first_doc.vector is not None and hasattr(first_doc.vector, '__len__'):
                    logger.info(f"   → 第一个文档向量长度: {len(first_doc.vector)}")

        all_conditions_met = self.use_vector_db and self.vector_db and transformed_docs
        logger.info(f"   → 所有条件是否满足: {all_conditions_met}")
        logger.info("=" * 80)

        if all_conditions_met:
            try:
                logger.info("=" * 80)
                logger.info("💾 [VECTOR DB] 保存文档到向量数据库")

                # 🔧 关键修复：使用 is_new_repository 判断，而不是 git_changes.is_initial
                # 因为 git_changes.is_initial 只是判断是否有 previous_hash，不代表数据库中是否有数据
                if not is_new_repository and not git_changes.is_initial and git_changes.changes:
                    # 增量更新：先删除变更文件的旧向量，再添加新向量
                    logger.info(f"   → 增量更新模式（已存在仓库）")

                    # 🔧 关键修复：先删除变更文件的所有向量，避免重复
                    modified_files = set()
                    deleted_files = set()

                    for change in git_changes.changes:
                        if change.change_type in [FileChangeType.MODIFIED, FileChangeType.ADDED]:
                            modified_files.add(change.file_path)
                        elif change.change_type == FileChangeType.DELETED:
                            deleted_files.add(change.file_path)

                    # 删除所有变更文件的向量
                    all_changed_files = modified_files | deleted_files
                    if all_changed_files:
                        logger.info(f"   → 物理删除 {len(all_changed_files)} 个变更文件的旧向量")
                        for file_path in all_changed_files:
                            count = self.vector_db.delete_by_metadata("file_path", file_path, soft_delete=False)
                            logger.debug(f"      - 删除 {file_path}: {count} 个文档")

                    # 找出需要重新添加的文档（新增或修改的文件）
                    new_docs = []
                    for file_path in modified_files:
                        matching_docs = [
                            doc for doc in transformed_docs
                            if doc.meta_data.get('file_path') == file_path
                        ]
                        new_docs.extend(matching_docs)

                    if new_docs:
                        logger.info(f"   → 添加 {len(new_docs)} 个新文档")
                        doc_ids = self.vector_db.add_documents(new_docs)
                        logger.info(f"   ✅ 成功添加 {len(doc_ids)} 个文档")
                    else:
                        logger.info(f"   → 无新文档需要添加")
                else:
                    # 全量更新（新建仓库或 Git 首次处理）
                    if is_new_repository:
                        logger.info(f"   → 全量更新模式（新建仓库）: {len(transformed_docs)} 个文档")
                    else:
                        logger.info(f"   → 全量更新模式（Git 首次处理）: {len(transformed_docs)} 个文档")

                    doc_ids = self.vector_db.add_documents(transformed_docs)
                    logger.info(f"   ✅ 成功保存 {len(doc_ids)} 个文档")

                # 验证保存
                if hasattr(self.vector_db, 'count_documents'):
                    count = self.vector_db.count_documents()
                    logger.info(f"   ✅ 验证: 数据库中当前文档总数: {count}")

                logger.info("🎉 [VECTOR DB] 文档保存完成")
                logger.info("=" * 80)

            except Exception as e:
                logger.error(f"❌ [VECTOR DB] 保存文档到向量数据库失败: {e}")
                logger.error(f"   → 错误类型: {type(e).__name__}")
                import traceback
                logger.error(f"   → 详细错误:\n{traceback.format_exc()}")
                logger.info("   → 将继续使用传统 LocalDB")

        logger.info("=" * 80)
        logger.info("✅ [GIT INCREMENTAL] Wiki 刷新完成")
        logger.info("=" * 80)

        # 🔧 关键修复：最终检查向量是否已保存到数据库
        if self.use_vector_db and self.vector_db and transformed_docs:
            logger.info("=" * 80)
            logger.info("🔍 [FINAL CHECK] 最终向量数据库检查")

            try:
                # 检查数据库中的文档数量
                if hasattr(self.vector_db, 'count_documents'):
                    db_count = self.vector_db.count_documents()
                    logger.info(f"   → 数据库中当前文档数: {db_count}")
                    logger.info(f"   → 内存中文档数: {len(transformed_docs)}")

                    # 如果数据库为空但内存有数据，说明保存失败了，需要重新保存
                    if db_count == 0 and len(transformed_docs) > 0:
                        logger.warning("   ⚠️ 数据库为空但内存有文档，尝试重新保存...")

                        # 检查第一个文档是否有向量
                        first_doc_has_vector = (
                            len(transformed_docs) > 0 and
                            hasattr(transformed_docs[0], 'vector') and
                            transformed_docs[0].vector is not None
                        )

                        logger.info(f"   → 第一个文档有向量: {first_doc_has_vector}")

                        if first_doc_has_vector:
                            logger.info(f"   → 正在保存 {len(transformed_docs)} 个文档到数据库...")
                            doc_ids = self.vector_db.add_documents(transformed_docs)
                            logger.info(f"   ✅ 成功保存 {len(doc_ids)} 个文档")

                            # 再次验证
                            db_count = self.vector_db.count_documents()
                            logger.info(f"   ✅ 验证: 数据库中现在有 {db_count} 个文档")
                        else:
                            logger.error("   ❌ 文档没有向量，无法保存到数据库")
                    elif db_count > 0:
                        logger.info(f"   ✅ 数据库已有文档，无需重新保存")
                    else:
                        logger.warning(f"   ⚠️ 内存和数据库都为空")

            except Exception as e:
                logger.error(f"   ❌ 最终检查失败: {e}")
                import traceback
                logger.error(f"   详细错误: {traceback.format_exc()}")

            logger.info("=" * 80)

        return transformed_docs

    def prepare_retriever(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_url_or_path, repo_type, access_token)
