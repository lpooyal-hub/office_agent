import logging
import re
from threading import RLock

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency for lightweight tests
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from services.auth import build_document_where_filter, normalize_allowed_roles
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

    def build_records_for_document(
        self,
        path,
        text=None,
        owner="system",
        visibility="public",
        allowed_roles=None,
    ):
        text = text if text is not None else read_document_text(path)
        allowed_roles = normalize_allowed_roles(allowed_roles)
        chunks = split_text(text)
        if not chunks:
            return []

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embedder().encode(chunk_texts, normalize_embeddings=True)
        records = []
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{path.name}::chunk::{chunk.chunk_index}",
                    "embedding": embeddings[index].tolist(),
                    "document": chunk.text,
                    "metadata": {
                        "stored_name": path.name,
                        "display_name": get_display_filename(path.name),
                        "section_title": chunk.section_title,
                        "chunk_index": chunk.chunk_index,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "source_kind": "document",
                        "owner": owner,
                        "visibility": visibility,
                        "allowed_roles": ",".join(allowed_roles),
                        "allowed_role_viewer": "viewer" in allowed_roles,
                        "allowed_role_editor": "editor" in allowed_roles,
                        "allowed_role_admin": "admin" in allowed_roles,
                    },
                }
            )
        return records

    def index_documents(self, paths, owner="system", visibility="public", allowed_roles=None):
        all_records = []
        for path in paths:
            all_records.extend(
                self.build_records_for_document(
                    path,
                    owner=owner,
                    visibility=visibility,
                    allowed_roles=allowed_roles,
                )
            )

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
                    roles = document.get("allowed_roles") or ["viewer", "editor", "admin"]
                    records.extend(
                        self.build_records_for_document(
                            path,
                            owner=document.get("owner", "system"),
                            visibility=document.get("visibility", "public"),
                            allowed_roles=roles,
                        )
                    )
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

    def get_relevant_rules(self, query, threshold=0.35, top_k=3, user=None):
        self.bootstrap_documents()

        query_embedding = self.get_embedder().encode([query], normalize_embeddings=True)[0]
        default_matches = self.get_default_rule_matches(query_embedding, query)
        document_matches = self.get_document_matches(query_embedding, user=user)
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

        ranked_matches = rank_matches(query, combined_matches)

        selected = []
        for match in ranked_matches:
            if match["semantic_score"] >= threshold or not selected:
                selected.append(match)
            if len(selected) >= top_k:
                break

        rendered = "\n\n".join(
            f"- {match['chunk']}\n  유사도: {match['semantic_score']:.2f} / "
            f"키워드: {match['lexical_score']:.2f} / 순위점수: {match['rank_score']:.2f}"
            for match in selected
        )
        return rendered, selected

    def get_default_rule_matches(self, query_embedding, query, limit=5):
        default_rule_embeddings = self.get_default_rule_embeddings()
        similarities = dot_similarities(default_rule_embeddings, query_embedding)
        lexical_scores = get_lexical_scores(query, self._default_rule_chunks)
        ranking_scores = [similarities[index] + lexical_scores[index] for index in range(len(similarities))]
        ranked_indexes = sorted(range(len(ranking_scores)), key=lambda index: ranking_scores[index], reverse=True)[:limit]

        matches = []
        for index in ranked_indexes:
            chunk = self._default_rule_chunks[index]
            matches.append(
                {
                    "chunk": chunk,
                    "semantic_score": float(similarities[index]),
                    "score": float(similarities[index]),
                    "lexical_score": 0.0,
                    "rank_score": float(similarities[index]),
                    "source_kind": "default_rule",
                    "chunk_index": int(index),
                }
            )
        return matches

    def get_document_matches(self, query_embedding, limit=8, user=None):
        where = build_document_where_filter(user) if user else None
        response = self.chroma_store.query(query_embedding.tolist(), n_results=limit, where=where)
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
            location = format_source_location(metadata)
            rendered_chunk = f"[{display_name}]"
            if location:
                rendered_chunk = f"{rendered_chunk} {location}"
            rendered_chunk = f"{rendered_chunk}\n{document_text}"
            matches.append(
                {
                    "chunk": rendered_chunk,
                    "semantic_score": score,
                    "score": score,
                    "lexical_score": 0.0,
                    "rank_score": score,
                    "source_kind": metadata.get("source_kind") or "document",
                    "chunk_index": metadata.get("chunk_index"),
                }
            )
        return matches


def format_source_location(metadata):
    section_title = metadata.get("section_title") or ""
    chunk_index = metadata.get("chunk_index")
    char_start = metadata.get("char_start")
    char_end = metadata.get("char_end")

    details = []
    if section_title:
        details.append(section_title)
    if chunk_index is not None:
        details.append(f"chunk {int(chunk_index) + 1}")
    if char_start is not None and char_end is not None:
        details.append(f"chars {int(char_start)}-{int(char_end)}")

    return f"({' · '.join(details)})" if details else ""
def dot_similarities(embeddings, query_embedding):
    if np is not None:
        return np.dot(embeddings, query_embedding)
    return [
        sum(float(value) * float(query_embedding[index]) for index, value in enumerate(embedding))
        for embedding in embeddings
    ]


def rank_matches(query, semantic_matches):
    """Return matches sorted by combined semantic and lexical relevance score."""
    if not semantic_matches:
        return []

    ranked = []
    lexical_scores = get_lexical_scores(query, [item.get("chunk", "") for item in semantic_matches])
    for index, match in enumerate(semantic_matches):
        semantic_score = float(match.get("semantic_score", match.get("score", 0.0)) or 0.0)
        lexical_score = float(lexical_scores[index])
        enriched = dict(match)
        enriched["semantic_score"] = semantic_score
        enriched["score"] = semantic_score
        enriched["lexical_score"] = lexical_score
        enriched["rank_score"] = semantic_score + lexical_score
        enriched.setdefault("source_kind", "default_rule")
        enriched.setdefault("chunk_index", None)
        ranked.append(enriched)

    return sorted(ranked, key=lambda item: item["rank_score"], reverse=True)


def convert_distance_to_score(distance):
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def parse_rule_match(match, score=None):
    if isinstance(match, dict):
        chunk_text = match.get("chunk", "")
        semantic_score = float(match.get("semantic_score", match.get("score", score or 0.0)) or 0.0)
        lexical_score = float(match.get("lexical_score", 0.0) or 0.0)
        rank_score = float(match.get("rank_score", semantic_score + lexical_score) or 0.0)
        source_kind = match.get("source_kind") or "default_rule"
        chunk_index = match.get("chunk_index")
    else:
        chunk_text = match
        semantic_score = float(score or 0.0)
        lexical_score = 0.0
        rank_score = semantic_score
        source_kind = "default_rule"
        chunk_index = None

    lines = chunk_text.splitlines()
    source = "기본 내규"
    excerpt = chunk_text
    if lines and lines[0].startswith("["):
        header_match = re.match(r"^\[([^\]]+)\]\s*(?:\((.*)\))?$", lines[0])
        if not header_match:
            header_match = re.match(r"^\[([^\]]+)\]", lines[0])
        source = header_match.group(1) if header_match else lines[0][1:-1]
        location = header_match.group(2) if header_match and header_match.lastindex and header_match.lastindex >= 2 else ""
        excerpt = "\n".join(lines[1:]).strip() or "(본문 없음)"

    return {
        "source": source,
        "score": round(score, 2),
        "location": locals().get("location", ""),
        "score": round(semantic_score, 2),
        "semantic_score": round(semantic_score, 4),
        "lexical_score": round(lexical_score, 4),
        "rank_score": round(rank_score, 4),
        "source_kind": source_kind,
        "chunk_index": chunk_index,
        "excerpt": excerpt[:320].strip(),
    }


def get_lexical_scores(query, chunks):
    keywords = expand_query_terms(query)
    if not keywords:
        return np.zeros(len(chunks)) if np is not None else [0.0] * len(chunks)

    scores = []
    for chunk in chunks:
        lowered = chunk.lower()
        matched = sum(1 for keyword in keywords if keyword in lowered)
        scores.append(min(matched * 0.35, 1.05))
    return np.array(scores) if np is not None else scores


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
