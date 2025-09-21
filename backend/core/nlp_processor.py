import re
from typing import List, Dict, Any

class NLPProcessor:
    def __init__(self):
        # Built-in minimal english stopwords to avoid external downloads
        self.stop_words = {
            'a','an','the','and','or','but','if','while','of','at','by','for','with','about','against',
            'between','into','through','during','before','after','above','below','to','from','up','down',
            'in','out','on','off','over','under','again','further','then','once','here','there','when',
            'where','why','how','all','any','both','each','few','more','most','other','some','such',
            'no','nor','not','only','own','same','so','than','too','very','can','will','just','don',
            'should','now','is','are','was','were','be','been','being','do','does','did','having','have',
            'has','i','you','he','she','it','we','they','me','him','her','them','my','your','his','its',
            'our','their','yours','ours','theirs'
        }
    
    def preprocess_text(self, text: str) -> str:
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize via regex and remove stopwords
        tokens = re.findall(r"[a-zA-Z]+", text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        # Simple regex-based entity-like extraction as a fallback
        entities: List[Dict[str, Any]] = []
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            entities.append({
                "text": match.group(0),
                "label": "PROPER_NOUN",
                "start": match.start(),
                "end": match.end()
            })
        return entities
    
    def extract_nouns(self, text: str) -> List[str]:
        # Heuristic noun-like extraction: words longer than 2 not in stopwords
        tokens = re.findall(r"[A-Za-z]+", text)
        nouns = [t for t in tokens if t.isalpha() and len(t) > 2 and t.lower() not in self.stop_words]
        return nouns
    
    def extract_adjectives(self, text: str) -> List[str]:
        # Heuristic adjective-like extraction using common adjective suffixes
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        adj_suffixes = ("y", "ful", "ous", "able", "ible", "ish", "less", "al", "ic")
        adjectives = [t for t in tokens if any(t.endswith(s) for s in adj_suffixes)]
        return adjectives