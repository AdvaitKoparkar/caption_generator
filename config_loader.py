"""
Configuration loader for the application.
Provides a robust interface for managing application settings.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ServiceConfig:
    host: str
    port: int
    model: Optional[str] = None  # Only for Ollama service

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class GiTConfig:
    checkpoint_dir: str
    run_name: str
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AppConfig:
    """
    Main configuration class that manages all application settings.
    Provides methods to load, save, and update configuration.
    """
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.services: Dict[str, ServiceConfig] = {}
        self.git_config: Optional[GiTConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            self._create_default_config()
            return

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                self._load_services(data.get('services', {}))
                self._load_git_config(data.get('GiT_checkpoint', {}))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")

    def _load_services(self, services_data: Dict[str, Any]) -> None:
        """Load service configurations."""
        self.services = {
            name: ServiceConfig(**config)
            for name, config in services_data.items()
        }

    def _load_git_config(self, git_data: Dict[str, Any]) -> None:
        """Load GiT configuration."""
        self.git_config = GiTConfig(**git_data)

    def _create_default_config(self) -> None:
        """Create default configuration if none exists."""
        default_config = {
            "services": {
                "main_app": {"host": "0.0.0.0", "port": 5000},
                "git_api": {"host": "0.0.0.0", "port": 8000},
                "ollama": {"host": "localhost", "port": 11434, "model": "llama2"}
            },
            "GiT_checkpoint": {
                "checkpoint_dir": "checkpoints",
                "run_name": "latest",
                "device": "auto"
            }
        }
        self._load_services(default_config['services'])
        self._load_git_config(default_config['GiT_checkpoint'])
        self.save()

    def save(self) -> None:
        """Save current configuration to JSON file."""
        config_data = {
            "services": {
                name: service.to_dict()
                for name, service in self.services.items()
            },
            "GiT_checkpoint": self.git_config.to_dict() if self.git_config else {}
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Error saving config: {e}")

    def update_service(self, service_name: str, **kwargs) -> None:
        """Update a service configuration."""
        if service_name not in self.services:
            raise KeyError(f"Service {service_name} not found")
        
        current_config = self.services[service_name].to_dict()
        current_config.update(kwargs)
        self.services[service_name] = ServiceConfig(**current_config)
        self.save()

    def update_git_config(self, **kwargs) -> None:
        """Update GiT configuration."""
        if not self.git_config:
            raise RuntimeError("GiT config not initialized")
        
        current_config = self.git_config.to_dict()
        current_config.update(kwargs)
        self.git_config = GiTConfig(**current_config)
        self.save()

    def get_service_url(self, service_name: str) -> str:
        """Get the full URL for a service."""
        if service_name not in self.services:
            raise KeyError(f"Service {service_name} not found")
        
        service = self.services[service_name]
        return f"http://{service.host}:{service.port}"

    def get_ollama_url(self) -> str:
        """Get the Ollama API URL."""
        return self.get_service_url('ollama')

    def get_git_api_url(self) -> str:
        """Get the GiT API URL."""
        return self.get_service_url('git_api')

    def get_main_app_url(self) -> str:
        """Get the main application URL."""
        return self.get_service_url('main_app')

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"AppConfig(services={list(self.services.keys())}, git_config={self.git_config})"

# Create a global config instance
config = AppConfig() 