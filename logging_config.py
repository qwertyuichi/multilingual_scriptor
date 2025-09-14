"""ロギング初期化ユーティリティ。

目的:
 - 統一フォーマット / レベル制御
 - コンソール + ローテーションファイル出力 (オプション)
 - config.toml の [logging] セクション (なければ default) を参照

使用方法:
    from logging_config import setup_logging, get_logger
    setup_logging(config_dict)
    logger = get_logger(__name__)

config.toml 例:
[logging]
level = "INFO"          # DEBUG / INFO / WARNING / ERROR
file_enabled = true
file_path = "app.log"
max_bytes = 1048576
backup_count = 3

"""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

_LOG_INITIALIZED = False

DEFAULT_LOG_CONF = {
    "level": "INFO",
    "file_enabled": False,
    "file_path": "app.log",
    "max_bytes": 1_000_000,
    "backup_count": 2,
}

LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"


def setup_logging(config: Dict[str, Any] | None = None) -> None:
    global _LOG_INITIALIZED
    if _LOG_INITIALIZED:
        return
    cfg = dict(DEFAULT_LOG_CONF)
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg})
    level = LEVEL_MAP.get(str(cfg.get("level", "INFO")).upper(), logging.INFO)

    logging.basicConfig(level=level, format=FORMAT, datefmt=DATEFMT)

    if cfg.get("file_enabled"):
        try:
            handler = RotatingFileHandler(
                filename=cfg.get("file_path", "app.log"),
                maxBytes=int(cfg.get("max_bytes", 1_000_000)),
                backupCount=int(cfg.get("backup_count", 2)),
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
            logging.getLogger().addHandler(handler)
        except Exception as e:
            logging.getLogger(__name__).warning(f"ファイルロギング初期化失敗: {e}")
    _LOG_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

__all__ = ["setup_logging", "get_logger"]
