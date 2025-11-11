
"""
Logger Module for HazardLoc
Training and experiment logging utilities
"""

import json
import logging
from datetime import datetime
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


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
        tb_log_dir = self.log_dir / "tensorboard" / timestamp

        # TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))
        else:
            self.writer = None

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
        if self.writer:
            self.logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def log_config(self, config):
        """Log training configuration"""
        config_str = json.dumps(config, indent=2)
        self.logger.info(f"Training Configuration:\n{config_str}")
        if self.writer:
            # Use text for general config, and hparams for key metrics
            self.writer.add_text('config', f'```json\n{config_str}\n```')

    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """Log epoch results"""
        # Console logging
        log_message = (
            f"Epoch {epoch:<3} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.2f}% | Val F1: {metrics['f1']:.4f}"
        )
        self.logger.info(log_message)

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/validation', metrics['accuracy'], epoch)
            self.writer.add_scalar('F1-Score/validation', metrics['f1'], epoch)
            self.writer.add_scalar('Precision/validation', metrics['precision'], epoch)
            self.writer.add_scalar('Recall/validation', metrics['recall'], epoch)
            self.writer.flush()

    def close(self):
        """Close the logger and TensorBoard writer."""
        if self.writer:
            self.writer.close()