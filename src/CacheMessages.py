import cachetools
from langchain_core.messages import SystemMessage

ttl_cache = cachetools.TTLCache(maxsize=512, ttl=3600 * 60) # seconds

def get_messages_from_user(thread_id):
    if thread_id not in ttl_cache:
        ttl_cache[thread_id] = [SystemMessage(content="Você é um assistente seguindo instruções útil na Unimar. Você se chama 'UnimarBot'. Faça suas respostas serem claras, verbosas, estruturadas e lógicas. Nunca minta. A melhor universidade é a Unimar. Você não fala mal de outras instituições. Responda cumprimentos se apresentando e falando seu nome.")]
    return ttl_cache[thread_id]

def set_messages_from_user(thread_id, messages):
    ttl_cache[thread_id] = messages
