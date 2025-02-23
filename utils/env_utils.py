import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv


def load_local_env():
    """
    Load environment variables from the .env.local file at the base of the repository.
    """
    base_path = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=base_path / '.env.local')


def get_env_variable(env_var_name: str):
    """Get environment variable from current environment.
    
    :param var_name: Name of the environment variable to load
    """

    env_var = os.getenv(env_var_name)
    if not env_var:
        raise ValueError(f"Environment variable {env_var_name} not found")

    return env_var