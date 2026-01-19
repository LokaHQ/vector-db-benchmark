import math
import re
from collections import Counter


class QueryGenerator:
    """Generates search queries from document text."""

    def __init__(self):
        """Initializes the query generator."""
        self.stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "been",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "would",
            "could",
            "should",
            "this",
            "these",
            "those",
            "they",
            "them",
            "their",
            "there",
            "where",
            "when",
            "what",
            "which",
            "who",
            "why",
            "how",
            "or",
            "but",
            "not",
            "no",
            "can",
            "may",
            "must",
            "shall",
            "have",
            "had",
            "do",
            "does",
            "did",
            "am",
            "were",
            "being",
            "any",
            "all",
            "some",
            "such",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "now",
            "here",
            "then",
            "once",
            "also",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "if",
            "because",
            "while",
            "until",
            "since",
        }
        self.max_text_length = 50000

    def generate_queries(self, document_text: str, num_queries: int = 3) -> list[str]:
        """Generates search queries from document text."""
        if not document_text or len(document_text.strip()) < 50:
            return []

        text = (
            document_text[: self.max_text_length]
            if len(document_text) > self.max_text_length
            else document_text
        )
        text = self._clean_text(text)

        key_phrases = self._extract_key_phrases(text)

        queries = []
        question_templates = [
            "What is {}?",
            "How does {} work?",
            "What are the benefits of {}?",
            "Can you explain {}?",
            "What is the purpose of {}?",
            "How to understand {}?",
            "What are the key features of {}?",
            "What is the significance of {}?",
        ]

        for i, phrase in enumerate(key_phrases[:num_queries]):
            template = question_templates[i % len(question_templates)]
            query = template.format(phrase)
            queries.append(query)

        return queries[:num_queries]

    def _clean_text(self, text: str) -> str:
        """Cleans and normalizes text for processing."""
        text = re.sub(r"\s+", " ", text.strip())
        lines = text.split("\n")
        meaningful_lines = [
            line.strip()
            for line in lines
            if len(line.strip()) > 20 and not line.strip().isupper()
        ]
        return " ".join(meaningful_lines)

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extracts key phrases using TF-IDF scoring."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        words = [w for w in words if w not in self.stopwords and len(w) > 2]

        phrases = []
        for i in range(len(words) - 1):
            if i < len(words) - 2:
                phrases.append(" ".join(words[i : i + 3]))
            phrases.append(" ".join(words[i : i + 2]))

        if not phrases:
            return []

        phrase_counts = Counter(phrases)

        scored_phrases = []
        total_phrases = len(phrases)

        for phrase, count in phrase_counts.items():
            if count >= 2:
                tf = count / total_phrases
                idf = math.log(total_phrases / count)
                score = tf * idf
                scored_phrases.append((phrase, score))

        scored_phrases.sort(key=lambda x: x[1], reverse=True)
        return [phrase for phrase, score in scored_phrases[:15]]
