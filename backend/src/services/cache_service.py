class CacheService:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value, ttl=None):
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        self.cache = {}

cache_service = CacheService()

