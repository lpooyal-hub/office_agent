import logging
import re
from threading import RLock

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - optional dependency
    cosine_similarity = None

from services.document_library import list_stored_documents, read_document_text, split_text


class DocumentRetriever:
    def __init__(self, document_folder, embedding_model, default_rules):
        self.document_folder = document_folder
        self.embedding_model = embedding_model
        self.default_rules = list(default_rules)
        self._embedder = None
        self._rule_chunks = None
        self._rule_embeddings = None
        self._lock = RLock()

    def get_embedder(self):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers 패키지가 설치되지 않았습니다.")

        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    logging.info("Embedding model loading: %s", self.embedding_model)
                    self._embedder = SentenceTransformer(self.embedding_model, device="cpu")
        return self._embedder

    def load_rule_chunks(self):
        chunks = list(self.default_rules)
        for document in reversed(list_stored_documents(self.document_folder)):
            path = self.document_folder / document["stored_name"]
            try:
                text = read_document_text(path)
            except Exception as exc:
                logging.warning("Document load failed: %s (%s)", path.name, exc)
                continue

            for chunk in split_text(text):
                chunks.append(f"[{document['display_name']}]\n{chunk}")
        return chunks

    def get_rule_chunks(self):
        if self._rule_chunks is None:
            with self._lock:
                if self._rule_chunks is None:
                    self._rule_chunks = self.load_rule_chunks()
        return self._rule_chunks

    def get_rule_embeddings(self):
        if self._rule_embeddings is None:
            with self._lock:
                if self._rule_embeddings is None:
                    self._rule_embeddings = self.get_embedder().encode(
                        self.get_rule_chunks(),
                        normalize_embeddings=True,
                    )
        return self._rule_embeddings

    def invalidate(self):
        with self._lock:
            self._rule_chunks = None
            self._rule_embeddings = None

    def get_relevant_rules(self, query, threshold=0.35, top_k=3):
        chunks = self.get_rule_chunks()
        if not chunks:
            return "관련된 내규를 찾을 수 없습니다.", []

        if cosine_similarity is None:
            raise RuntimeError("scikit-learn 패키지가 설치되지 않았습니다.")

        query_vector = self.get_embedder().encode([query], normalize_embeddings=True)
        scores = cosine_similarity(query_vector, self.get_rule_embeddings())[0]
        lexical_scores = get_lexical_scores(query, chunks)
        combined_scores = scores + lexical_scores
        ranked_indexes = np.argsort(combined_scores)[::-1][:top_k]

        matches = []
        for index in ranked_indexes:
            score = float(scores[index])
            if score >= threshold or not matches:
                matches.append((chunks[index], score))

        rendered = "\n\n".join(
            f"- {document}\n  유사도: {score:.2f}" for document, score in matches
        )
        return rendered, matches


def parse_rule_match(chunk_text, score):
    lines = chunk_text.splitlines()
    source = "기본 내규"
    excerpt = chunk_text
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        source = lines[0][1:-1]
        excerpt = "\n".join(lines[1:]).strip() or "(본문 없음)"

    return {
        "source": source,
        "score": round(score, 2),
        "excerpt": excerpt[:320].strip(),
    }


def get_lexical_scores(query, chunks):
    keywords = expand_query_terms(query)
    if not keywords:
        return np.zeros(len(chunks))

    scores = []
    for chunk in chunks:
        lowered = chunk.lower()
        matched = sum(1 for keyword in keywords if keyword in lowered)
        scores.append(min(matched * 0.35, 1.05))
    return np.array(scores)


def expand_query_terms(query):
    particles = (
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "도",
        "만",
        "과",
        "와",
        "에서",
        "에게",
        "으로",
        "로",
        "입니다",
        "인가요",
        "나요",
        "요",
    )
    terms = set()

    for token in re.findall(r"[가-힣A-Za-z0-9]+", query.lower()):
        if len(token) < 2:
            continue

        terms.add(token)
        for particle in particles:
            if token.endswith(particle) and len(token) > len(particle) + 1:
                terms.add(token[: -len(particle)])

        if re.search(r"[가-힣]", token) and len(token) >= 3:
            terms.add(token[:2])

    return terms
