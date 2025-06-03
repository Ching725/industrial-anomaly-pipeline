# src/utils/logger.py
import os
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_dir="artifacts/logs", prefix="log"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.txt")
        self.terminal = sys.stdout
        self.log = open(self.log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def enable_logging(log_dir="artifacts/logs", prefix="log"):
    sys.stdout = Logger(log_dir=log_dir, prefix=prefix)