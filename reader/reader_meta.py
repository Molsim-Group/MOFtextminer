from abc import ABCMeta, abstractmethod


class Reader(metaclass=ABCMeta):
    @property
    @abstractmethod
    def suffix(self):
        pass

    @classmethod
    def check_suffix(cls, suffix):
        return suffix == cls.suffix

    @classmethod
    @abstractmethod
    def parsing(cls, file):
        pass

    @classmethod
    @abstractmethod
    def get_metadata(cls, file):
        pass
