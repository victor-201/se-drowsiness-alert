"""Driver drowsiness detection package."""


__all__ = [
    "AlertSystem",
    "Config",
    "DrowsinessDetector",
    "FacialAnalyzer",
    "ModelManager",
    "Settings",
    "run_calibration",
    "run_detection",
    "run_kivy",
]


def __getattr__(name):
    if name in {"run_kivy", "run_detection", "run_calibration"}:
        from importlib import import_module

        main = import_module("src.main")
        return getattr(main, name)
    if name == "DrowsinessDetector":
        from src.core.detector import DrowsinessDetector

        return DrowsinessDetector
    if name == "FacialAnalyzer":
        from src.core.facial_analyzer import FacialAnalyzer

        return FacialAnalyzer
    if name == "AlertSystem":
        from src.core.alert_system import AlertSystem

        return AlertSystem
    if name == "ModelManager":
        from src.core.model_manager import ModelManager

        return ModelManager
    if name == "Config":
        from src.configs.config import Config

        return Config
    if name == "Settings":
        from src.configs.settings import Settings

        return Settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
