"""
Core configuration settings for Quantro Trading Platform
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Application
    DEBUG: bool = False
    SECRET_KEY: str
    ALLOWED_HOSTS: str = "*"
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30

    # External APIs
    CCXT_SANDBOX: bool = True
    SET_API_KEY: str | None = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/quantro.log"

    model_config = {"env_file": ".env", "case_sensitive": True}

    @property
    def allowed_hosts_list(self) -> list[str]:
        """Parse ALLOWED_HOSTS string into a list"""
        if isinstance(self.ALLOWED_HOSTS, str):
            return [host.strip() for host in self.ALLOWED_HOSTS.split(",")]
        return self.ALLOWED_HOSTS


# Global settings instance
settings = Settings()
