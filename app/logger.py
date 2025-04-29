import logging

class AppLogger:
    @staticmethod
    def get_logger(name: str = __name__) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        return logging.getLogger(name)
