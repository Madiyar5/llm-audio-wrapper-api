from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
class Settings(BaseSettings):
    app_name: str = "LLM Audio Wrapper API"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    log_level: str = "INFO"
 
    ollama_base_url: str = "http://ollama:11434"
    model_name: str = "qwen2.5:14b"
    ollama_timeout: int = 180
 
    whisper_model_size: str = "small"
    whisper_compute_type: str = "int8"
    upload_dir: str = "uploads"
 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
 
 
settings = Settings()

