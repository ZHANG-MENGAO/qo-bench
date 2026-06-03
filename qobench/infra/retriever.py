# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""LangChain BaseRetriever wrapping pymilvus hybrid_search.

Why custom: langchain_milvus.Milvus does not expose group_by_field +
RRF-pinned hybrid in a clean way. We wrap pymilvus directly and emit
LangChain Documents.

Caller pattern:
    retriever = build_retriever(embedding_fn)
    retriever.current_w_start = "2015-03-01"
    retriever.current_w_end = "2015-03-31"
    docs = retriever.invoke("M&A rumor")
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from qobench import config
from qobench.infra.date_filter import build_date_expr
from qobench.infra.reranker import rerank_sync
from qobench.infra.whitelist import load_v2_whitelist

OUTPUT_FIELDS = [
    "chunk_text",
    "url_hash",
    "date",
    "tickers",
    "title",
    "publisher",
    "chunk_index",
    "total_chunks",
    "url",   # Milvus field name; renamed to article_url in doc metadata below
]


class MilvusHybridRetriever(BaseRetriever):
    """Hybrid (BM25 + dense) Milvus retriever with date-window filtering."""

    client: Any
    collection_name: str
    vector_field: str
    sparse_field: str | None
    date_field: str
    date_format: str
    url_hash_field: str
    embedding_fn: Any  # callable: str -> list[float]
    top_k_initial: int = config.TOP_K_INITIAL
    top_k_reranked: int = config.TOP_K_RERANKED
    rerank_api_key: str | None = None
    rrf_k: int = config.RRF_K
    pad_days: int = config.DATE_WINDOW_PAD_DAYS

    # Mutable per-call; set by the runner before each invoke()
    current_w_start: str | None = None
    current_w_end: str | None = None
    current_ticker: str | None = None  # F3: ticker pre-filter for T02/T26 (params.F)

    # Cross-round dedup state: ReAct populates this with url_hashes from prior
    # search_news rounds so round N+1 excludes already-seen articles. Naive RAG
    # ignores it (always empty). pydantic-v2 BaseRetriever deep-copies the
    # default per instance, so the literal set() is safe here.
    seen_url_hashes: set[str] = set()

    # V2 bundle canonical substrate filter — restricts retrieval to the 22,984
    # articles in eval_bundle_v2_2026-05-14/corpus/articles.csv. Loaded once via
    # infra.whitelist.load_v2_whitelist() and passed in by the factories. None =
    # no filter (full fnspid_articles, ablation only). See whitelist.py for
    # rationale and design decision (2026-05-14) quote.
    url_hash_whitelist: frozenset[str] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        if not (self.current_w_start and self.current_w_end):
            raise RuntimeError(
                "Set current_w_start / current_w_end before calling invoke()"
            )

        date_expr = build_date_expr(
            self.date_field,
            self.current_w_start,
            self.current_w_end,
            pad_days=self.pad_days,
            date_format=self.date_format,
        )
        # F5 was a `text_length > 200` cutoff added to drop title-only
        # stubs. the upstream upstream pipeline now removes those (probe on
        # 2026-05-08: 0 chunks with text_length <= 50, 94 / 494,841
        # = 0.019% with text_length <= 200). The remaining short rows are
        # legit syndicated content — let RRF handle them.
        expr_parts = [date_expr]
        # F3: ticker pre-filter when the question pinned a firm (T02 / T26).
        if self.current_ticker:
            safe = self.current_ticker.replace('"', '\\"')
            expr_parts.append(f'array_contains(tickers, "{safe}")')
        # Cross-round dedup (ReAct only): exclude url_hashes seen in prior rounds.
        if self.seen_url_hashes:
            quoted = ", ".join(f'"{h}"' for h in sorted(self.seen_url_hashes))
            expr_parts.append(f"url_hash not in [{quoted}]")
        # V2 canonical-substrate filter: restrict to the 22,984 articles the v2
        # bundle defines as GT scope. ~808 KB filter expression, Milvus handles
        # it in <0.5s on hybrid_search (measured 2026-05-14).
        if self.url_hash_whitelist:
            quoted = ", ".join(f'"{h}"' for h in self.url_hash_whitelist)
            expr_parts.append(f"url_hash in [{quoted}]")
        full_expr = " and ".join(expr_parts)

        dense_vec = self.embedding_fn(query)

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field=self.vector_field,
            param={"metric_type": "COSINE"},
            limit=self.top_k_initial,
            expr=full_expr,
        )

        reqs = [dense_req]
        if self.sparse_field:
            sparse_req = AnnSearchRequest(
                data=[query],
                anns_field=self.sparse_field,
                param={"metric_type": "BM25"},
                limit=self.top_k_initial,
                expr=full_expr,
            )
            reqs.append(sparse_req)

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=RRFRanker(self.rrf_k),
            limit=self.top_k_initial,
            group_by_field=self.url_hash_field,
            output_fields=OUTPUT_FIELDS,
        )

        # Materialize the initial top-N hits into Documents first.
        # Rename Milvus's `url` field → `article_url` in doc metadata so the
        # rest of the pipeline (cited_urls in the LLM schema, tool output
        # formatting) reads a consistent name.
        docs: list[Document] = []
        for hit in results[0]:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else hit.entity
            text = entity.get("chunk_text", "") or ""
            metadata = {k: v for k, v in entity.items() if k != "chunk_text"}
            if "url" in metadata:
                metadata["article_url"] = metadata.pop("url")
            docs.append(Document(page_content=text, metadata=metadata))

        if not docs:
            return docs

        # Rerank step: top_k_initial → top_k_reranked. Errors propagate — no
        # silent fallback to unranked retrieval (spec §5.3).
        if self.rerank_api_key is None:
            raise RuntimeError(
                "MilvusHybridRetriever needs rerank_api_key; pass it from the caller."
            )
        scores = rerank_sync(query, [d.page_content for d in docs], self.rerank_api_key)
        ranked = sorted(zip(scores, range(len(docs))), key=lambda x: -x[0])
        keep_idx = [i for _, i in ranked[: self.top_k_reranked]]
        kept = [docs[i] for i in keep_idx]
        # Record returned hashes so the next round's filter excludes them.
        for d in kept:
            h = d.metadata.get("url_hash")
            if h:
                self.seen_url_hashes.add(h)
        return kept


def build_retriever(embedding_fn) -> MilvusHybridRetriever:
    """Factory: read runtime config and return a wired retriever.

    Loads the v2 canonical url_hash whitelist by default, restricting retrieval
    to the 22,984-article benchmark substrate. Set QOBENCH_DISABLE_V2_WHITELIST=1
    to skip and run against the full fnspid_articles collection (ablation only).
    """
    rt = config.load_runtime()
    client = MilvusClient(
        uri=config.MILVUS_URI,
        token=config.MILVUS_TOKEN,
        db_name=config.MILVUS_DB_NAME,
    )
    return MilvusHybridRetriever(
        client=client,
        collection_name=rt["collection_name"],
        vector_field=rt["vector_field"],
        sparse_field=rt.get("sparse_field"),
        date_field=rt["date_field"],
        date_format=rt["date_format"],
        url_hash_field=rt["url_hash_field"],
        embedding_fn=embedding_fn,
        rerank_api_key=config.load_deepinfra_key(),
        url_hash_whitelist=load_v2_whitelist(),
    )
