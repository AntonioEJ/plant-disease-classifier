"""Logger estructurado para todos los módulos del proyecto."""
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Crea un logger estructurado con formato consistente.

    Args:
        name: Nombre del logger (usar __name__ en cada módulo)
        level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Ruta opcional a archivo de log

    Returns:
        Logger configurado

    Example:
        logger = get_logger(__name__)
        logger.info("Iniciando entrenamiento...")
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si el logger ya fue configurado
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler de archivo (opcional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
