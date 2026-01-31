"""
Configuration loader utility
"""

import os
from typing import Any, Dict, Optional
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for the traffic signal system.
    
    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Default values
    """
    
    DEFAULT_CONFIG = {
        "system": {
            "name": "Adaptive Traffic Signal Controller",
            "version": "1.0.0",
            "debug": False,
            "log_level": "INFO",
        },
        "perception": {
            "model": "yolov8n",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "device": "cpu",
        },
        "tracking": {
            "algorithm": "sort",
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
        },
        "queue_estimation": {
            "vehicle_length_meters": 4.5,
            "meters_per_pixel": 0.1,
        },
        "safety": {
            "min_crossing_time_per_meter": 1.5,
            "high_traffic_pedestrian_count": 10,
        },
        "emergency": {
            "detection_confidence": 0.7,
            "green_corridor_duration": 15.0,
            "cooldown_period": 10.0,
        },
        "rl_agent": {
            "algorithm": "DQN",
            "learning_rate": 0.0001,
            "gamma": 0.99,
        },
        "edge": {
            "target_fps": 15,
            "max_latency_ms": 100,
            "use_onnx": True,
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # Load default config
        self._load_defaults()
        
        # Load from file if provided
        if config_path:
            self._load_from_file(config_path)
        
        # Apply environment overrides
        self._apply_env_overrides()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.config = self._deep_copy(self.DEFAULT_CONFIG)
    
    def _load_from_file(self, path: str):
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            logger.warning(f"Configuration file not found: {path}")
            return
        
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot load config file")
            return
        
        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._merge_config(self.config, file_config)
                logger.info(f"Configuration loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Format: TRAFFIC_SECTION_KEY=value
        prefix = "TRAFFIC_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split("_")
                if len(parts) >= 2:
                    section = parts[0]
                    setting = "_".join(parts[1:])
                    
                    if section in self.config:
                        # Try to parse value type
                        parsed_value = self._parse_env_value(value)
                        self.config[section][setting] = parsed_value
                        logger.debug(f"Config override: {section}.{setting} = {parsed_value}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        return value
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge override into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a nested dictionary"""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'perception.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        parts = key.split(".")
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        parts = key.split(".")
        config = self.config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Output path (uses original path if None)
        """
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot save config")
            return
        
        save_path = path or self.config_path
        if not save_path:
            logger.warning("No save path specified")
            return
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def to_dict(self) -> Dict:
        """Get full configuration as dictionary"""
        return self._deep_copy(self.config)
