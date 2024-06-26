import logging

class MLRobotUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_message(self, message, log_text_widget, is_debug_mode):
        if is_debug_mode:
            self.logger.debug(message)
        else:
            self.logger.info(message)
        log_text_widget.insert('end', f"{message}\n")
        log_text_widget.see('end')
