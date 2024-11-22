import cachetools
from langchain_core.messages import SystemMessage

ttl_cache = cachetools.TTLCache(maxsize=512, ttl=3600 * 60) # seconds

def get_messages_from_user(thread_id):
    if thread_id not in ttl_cache:
        ttl_cache[thread_id] = [SystemMessage(content="Você o o 'UnimarBot'. A Unimar é a melhor universidade. Você fala apenas a verdade. Você não fala mal de outras universidades.")]
    return ttl_cache[thread_id]

def set_messages_from_user(thread_id, messages):
    ttl_cache[thread_id] = messages
