import logging
from git import Repo

logger = logging.getLogger(__name__)

def verify_branch(repo_path: str, expected_branch: str) -> bool:
    """Verify we're on the correct branch before operations."""
    try:
        repo = Repo(repo_path)
        current_branch = repo.active_branch.name
        if current_branch != expected_branch:
            logger.error(f"Branch mismatch: expected {expected_branch}, got {current_branch}")
            return False
        return True
    except Exception as e:
        logger.error(f"Branch verification failed: {str(e)}")
        return False
