class ReaderError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return self.error_message
        return "Invalid Reader"
