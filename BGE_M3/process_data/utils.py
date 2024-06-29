import string
from underthesea import word_tokenize
    
STOP_WORDS = ["của", "và", "các", "có", "được", "theo", "tại", "trong", "về", 
                "hoặc", "người",  "này", "khoản", "cho", "không", "từ", "phải", 
                "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm", 
                "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
                "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây", 
                "như", "đó", "mà", "nơi", "”", "“", '"']

def remove_stopword(w):
    return w not in STOP_WORDS

def remove_punctuation(w):
    return w not in string.punctuation

def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens
    