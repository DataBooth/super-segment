from pathlib import Path
from typing import Optional, List, Any, Dict, Union
import os
import tomllib 
from loguru import logger
from dotenv import load_dotenv, find_dotenv

class ProjectConfig:
    """
    ProjectConfig loads and manages configuration from multiple TOML files in a directory,
    supports environment variable loading from a .env file, and provides convenient access
    to both config values and environment variables.

    Features:
    - Recursively finds the project root by searching for a marker file (e.g., pyproject.toml)
    - Loads all TOML files in a specified directory, or a provided list of TOML files
    - Loads environment variables from a .env file using python-dotenv
    - Provides safe access to both config values and environment variables
    - Logs all key steps using Loguru
    """

    def __init__(
        self,
        conf_dir: str = "conf",
        toml_files: Optional[List[Union[str, Path]]] = None,
        project_root_marker: str = "pyproject.toml",
        start_dir: Optional[Union[str, Path]] = None,
        dotenv_file: str = ".env",
    ) -> None:
        """
        Initialises the ProjectConfig instance.

        Args:
            conf_dir: Directory containing TOML config files (relative to project root).
            toml_files: Optional list of TOML files to load (relative to conf_dir).
            project_root_marker: File name used to identify the project root.
            start_dir: Optional starting directory for project root search.
            dotenv_file: Name of the .env file to load environment variables from.
        """
        logger.info("Initialising ProjectConfig")
        self.project_root: Path = self.find_project_root(marker=project_root_marker, start_dir=start_dir)
        logger.info(f"Project root found: {self.project_root}")

        self.conf_dir: Path = (self.project_root / conf_dir).resolve()
        logger.info(f"Config directory set to: {self.conf_dir}")

        if toml_files is not None:
            self.toml_files: List[Path] = [self.conf_dir / Path(f) for f in toml_files]
        else:
            self.toml_files: List[Path] = sorted(self.conf_dir.glob("*.toml"))
        logger.info(f"TOML files to load: {[str(f) for f in self.toml_files]}")

        self.configs: Dict[str, dict] = self._load_all()
        self._load_dotenv(dotenv_file=dotenv_file)

    @staticmethod
    def find_project_root(
        marker: str = "pyproject.toml",
        start_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Recursively search parent directories for the project root marker file.

        Args:
            marker: The file name to identify the project root (e.g., 'pyproject.toml').
            start_dir: Directory to start searching from (defaults to current file's parent).

        Returns:
            Path to the project root directory.

        Raises:
            FileNotFoundError: If the marker file is not found in any parent directory.
        """
        current = Path(start_dir).resolve() if start_dir else Path(__file__).resolve().parent
        logger.debug(f"Searching for project root from: {current}")
        while current != current.parent:
            logger.debug(f"Checking for {marker} in {current}")
            if (current / marker).exists():
                logger.success(f"Found project root at: {current}")
                return current
            current = current.parent
        logger.error(f"Could not find project root (missing {marker})")
        raise FileNotFoundError(f"Could not find project root (missing {marker})")

    def _load_all(self) -> Dict[str, dict]:
        """
        Load all specified TOML configuration files.

        Returns:
            Dictionary mapping file stems to their loaded TOML content.
        """
        configs: Dict[str, dict] = {}
        for f in self.toml_files:
            logger.info(f"Loading TOML config: {f}")
            with open(f, "rb") as fp:
                configs[f.stem] = tomllib.load(fp)
        logger.success("All TOML configs loaded successfully")
        return configs

    def _load_dotenv(self, dotenv_file: str) -> None:
        """
        Load environment variables from a .env file using python-dotenv.

        Args:
            dotenv_file: Name of the .env file to load (relative to project root).
        """
        env_path = find_dotenv(str(self.project_root / dotenv_file))
        if env_path:
            logger.info(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
            logger.success("Environment variables loaded from .env")
        else:
            logger.warning(f"No .env file found at {self.project_root / dotenv_file}")

    def get(
        self,
        *keys: str,
        file: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Retrieve a value from the loaded TOML configs.

        Args:
            *keys: Sequence of keys for nested lookup.
            file: Optional file stem to restrict search to a specific config file.
            default: Value to return if the key path is not found.

        Returns:
            The value found, or the default if not found.
        """
        logger.debug(f"Getting config value: keys={keys}, file={file}, default={default}")
        sources = [file] if file else self.configs.keys()
        for src in sources:
            conf = self.configs.get(src, {})
            val = conf
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    val = None
                    break
            if val is not None:
                logger.info(f"Found value for {keys} in {src}: {val}")
                return val
        logger.warning(f"Config value {keys} not found, returning default: {default}")
        return default

    @staticmethod
    def get_env(key: str, default: Any = None) -> Optional[str]:
        """
        Safely access an environment variable, with optional default.

        Args:
            key: The environment variable name.
            default: Value to return if the variable is not set.

        Returns:
            The environment variable value, or the default if not set.
        """
        value = os.getenv(key, default)
        if value is not None:
            logger.info(f"Environment variable '{key}' found: {value}")
        else:
            logger.warning(f"Environment variable '{key}' not found, using default: {default}")
        return value

    def as_dict(self) -> Dict[str, dict]:
        """
        Return all loaded configuration as a dictionary.

        Returns:
            Dictionary of all loaded configs, keyed by file stem.
        """
        return dict(self.configs)

    def debug_print(self) -> None:
        """
        Pretty-print the loaded configuration for debugging purposes.
        """
        import pprint
        pprint.pprint(self.as_dict())

if __name__ == "__main__":
    config = ProjectConfig()
    config.debug_print()

    print("\nApp title:", config.get("app", "title", file="app_config"))
    print("UI sidebar min age:", config.get("sidebar", "age", "min", file="ui"))
    print("DB_HOST from env:", config.get_env("DB_HOST", default="localhost"))
    # print(f"\nExample token: {config.get_env('LOGFIRE_TOKEN')}")