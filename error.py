from reader.error import ReaderError


class MerError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return self.error_message
        return "Material Entity Recognition Error"


class MofError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return self.error_message
        return "MOF Error"


class DatabaseError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return self.error_message
        return "Database Error"


class MerWarning(Warning):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return self.error_message
        return "Material Entity Recognition Warning"
