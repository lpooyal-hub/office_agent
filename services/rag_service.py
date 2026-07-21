import logging
import re
from threading import RLock

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from services.document_library import (
    get_display_filename,
    list_stored_documents,
    read_document_text,
    split_text,
)


class DocumentRetriever:
    def __init__(self, document_folder, embedding_model, default_rules, chroma_store):
        self.document_folder = document_folder
        self.embedding_model = embedding_model
        self.default_rules = list(default_rules)
        self.chroma_store = chroma_store
        self._embedder = None
        self._default_rule_embeddings = None
        self._default_rule_chunks = list(default_rules)
        self._bootstrapped = False
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

    def get_default_rule_embeddings(self):
        if self._default_rule_embeddings is None:
            with self._lock:
                if self._default_rule_embeddings is None:
                    self._default_rule_embeddings = self.get_embedder().encode(
                        self._default_rule_chunks,
                        normalize_embeddings=True,
                    )
        return self._default_rule_embeddings

    def invalidate(self):
        self._bootstrapped = False

    def build_records_for_document(self, path, text=None):
        text = text if text is not None else read_document_text(path)
        chunks = split_text(text)
        if not chunks:
            return []

        embeddings = self.get_embedder().encode(chunks, normalize_embeddings=True)
        records = []
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{path.name}::chunk::{index}",
                    "embedding": embeddings[index].tolist(),
                    "document": chunk,
                    "metadata": {
                        "stored_name": path.name,
                        "display_name": get_display_filename(path.name),
                        "chunk_index": index,
                        "source_kind": "document",
                    },
                }
            )
        return records

    def index_documents(self, paths):
        all_records = []
        for path in paths:
            all_records.extend(self.build_records_for_document(path))

        self.chroma_store.upsert_records(all_records)
        self._bootstrapped = True

    def remove_document(self, stored_name):
        self.chroma_store.delete_by_where({"stored_name": stored_name})

    def clear_documents(self):
        self.chroma_store.clear_collection()
        self._bootstrapped = True

    def bootstrap_documents(self):
        if self._bootstrapped:
            return

        with self._lock:
            if self._bootstrapped:
                return

            records = []
            for document in list_stored_documents(self.document_folder):
                path = self.document_folder / document["stored_name"]
                try:
                    records.extend(self.build_records_for_document(path))
                except Exception as exc:
                    logging.warning("Document bootstrap failed: %s (%s)", path.name, exc)

            if records:
                self.chroma_store.upsert_records(records)
            self._bootstrapped = True

    def get_rule_chunk_count(self):
        chunk_count = len(self.default_rules)
        for document in list_stored_documents(self.document_folder):
            path = self.document_folder / document["stored_name"]
            try:
                chunk_count += len(split_text(read_document_text(path)))
            except Exception as exc:
                logging.warning("Document chunk count failed: %s (%s)", path.name, exc)
        return chunk_count

    def get_relevant_rules(self, query, threshold=0.35, top_k=3):
        query_embedding = self.get_embedder().encode([query], normalize_embeddings=True)[0]
        default_matches = self.get_default_rule_matches(query_embedding, query)

        try:
            self.bootstrap_documents()
            document_matches = self.get_document_matches(query_embedding)
        except Exception as exc:
            logging.warning("ChromaDB document retrieval failed; using default rules only: %s", exc)
            document_matches = []

        combined_matches = default_matches + document_matches

        if not combined_matches:
            return "관련된 내규를 찾을 수 없습니다.", []

        lexical_scores = get_lexical_scores(query, [item["chunk"] for item in combined_matches])
        for index, lexical_score in enumerate(lexical_scores):
            combined_matches[index]["rank_score"] = combined_matches[index]["score"] + float(lexical_score)

        ranked_matches = sorted(
            combined_matches,
            key=lambda item: item["rank_score"],
            reverse=True,
        )

        selected = []
        for match in ranked_matches:
            if match["score"] >= threshold or not selected:
                selected.append((match["chunk"], match["score"]))
            if len(selected) >= top_k:
                break

        rendered = "\n\n".join(
            f"- {document}\n  유사도: {score:.2f}" for document, score in selected
        )
        return rendered, selected

    def get_default_rule_matches(self, query_embedding, query, limit=5):
        default_rule_embeddings = self.get_default_rule_embeddings()
        similarities = np.dot(default_rule_embeddings, query_embedding)
        lexical_scores = get_lexical_scores(query, self._default_rule_chunks)
        ranked_indexes = np.argsort(similarities + lexical_scores)[::-1][:limit]

        matches = []
        for index in ranked_indexes:
            chunk = self._default_rule_chunks[index]
            matches.append(
                {
                    "chunk": chunk,
                    "score": float(similarities[index]),
                    "rank_score": float(similarities[index]),
                }
            )
        return matches

    def get_document_matches(self, query_embedding, limit=8):
        response = self.chroma_store.query(query_embedding.tolist(), n_results=limit)
        ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        distances = response.get("distances") or []

        if not ids or not ids[0]:
            return []

        matches = []
        for index, _ in enumerate(ids[0]):
            metadata = (metadatas[0][index] if metadatas and metadatas[0] else {}) or {}
            document_text = documents[0][index] if documents and documents[0] else ""
            distance = distances[0][index] if distances and distances[0] else None
            score = convert_distance_to_score(distance)
            display_name = metadata.get("display_name") or metadata.get("stored_name") or "업로드 문서"
            rendered_chunk = f"[{display_name}]\n{document_text}"
            matches.append(
                {
                    "chunk": rendered_chunk,
                    "score": score,
                    "rank_score": score,
                }
            )
        return matches


def convert_distance_to_score(distance):
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


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
