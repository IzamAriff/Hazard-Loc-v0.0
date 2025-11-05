
"""
Logger Module for HazardLoc
Training and experiment logging utilities
"""

import json
import logging
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """
    Logger for training experiments
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"

        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training logger initialized. Log file: {log_file}")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def log_config(self, config):
        """Log training configuration"""
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """Log epoch results"""
        self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.2f}%, F1: {metrics['f1']:.4f}")