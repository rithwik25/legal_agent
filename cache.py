import hashlib

# Caching mechanism for repeated queries
class QueryCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size

    def get(self, query: str) -> str:
        """Retrieve cached response for a query."""
        # Use a hash of the query to create a consistent key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.cache.get(query_hash)

    def set(self, query: str, response: str):
        """Cache a response for a query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # If cache is full, remove the oldest entry
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[query_hash] = response