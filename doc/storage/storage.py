import numpy as np
import regex

from error import DatabaseError


class DataStorage(object):
    """Chemical/small molecule/ion/unit storage"""
    def __init__(self, storage_name, hash_key):
        """
        Database in doc.Document
        :param storage_name: (str) name of storage
        :param hash_key: (1 str) key of hash
        """
        self.data_to_hash = {}
        self.hash_to_data = {}
        self.name = storage_name
        self.hash_key = hash_key

    def _is_hash(self, item):
        if regex.match(fr'-{self.hash_key}\d\d\d\d\d\d-', item):
            return 'hash'
        else:
            return None

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.data_to_hash)

    def __getitem__(self, item):
        if item in self.data_to_hash:
            return self.data_to_hash[item]
        elif item in self.hash_to_data:
            return self.hash_to_data[item]
        else:
            raise DatabaseError(f'{item} not in {self.name} storage')

    def __contains__(self, item):
        return item in self.data_to_hash or item in self.hash_to_data

    def get(self, k, d=None):
        try:
            return self[k]
        except DatabaseError:
            return d

    def _generate_hash(self):
        for _ in range(100000):
            num = np.random.randint(1000000)
            num = str(num).zfill(6)
            hashcode = f'-{self.hash_key}{num}-'
            if hashcode not in self.hash_to_data:
                return hashcode
        raise DatabaseError('Hashcode cannot be specified')

    def append(self, item, hashcode=None):
        if item in self.data_to_hash:  # already exist
            return self.data_to_hash[item]
        else:
            if not hashcode:
                hashcode = self._generate_hash()
            self.data_to_hash[item] = hashcode

            if hashcode not in self.hash_to_data:
                self.hash_to_data[hashcode] = item
            return hashcode

    def pop(self, item):
        if item not in self:
            raise DatabaseError(f'{item} not in {self.name} storage')

        item2 = self.__getitem__(item)
        if self._is_hash(item):
            self.data_to_hash.pop(item2)
            self.hash_to_data.pop(item)
        else:
            self.data_to_hash.pop(item)
            self.hash_to_data.pop(item2)

        return item2

    @classmethod
    def from_file(cls, file_name, storage_name, hash_key):
        pass

    def keys(self):
        return self.data_to_hash.keys()

    def values(self):
        return self.data_to_hash.values()

    def items(self):
        return self.data_to_hash.items()