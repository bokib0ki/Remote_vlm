import os


OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

GEMINI_BASE_URL = os.environ.get('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

DEFAULT_PROVIDER = os.environ.get('SOTA_PROVIDER', '')
DEFAULT_MODEL = os.environ.get('SOTA_MODEL', '')
DEFAULT_MAX_NEW = int(os.environ.get('SOTA_MAX_NEW', '512'))
DEFAULT_TIMEOUT = int(os.environ.get('SOTA_TIMEOUT', '180'))
