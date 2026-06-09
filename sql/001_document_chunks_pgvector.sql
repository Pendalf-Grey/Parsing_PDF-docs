CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id text PRIMARY KEY,
    doc_id text NOT NULL,
    source_file text NOT NULL,
    chunk_index integer NOT NULL,
    page_start integer,
    page_end integer,
    section_path text[] NOT NULL DEFAULT '{}',
    text text NOT NULL,
    word_count integer NOT NULL,
    source_record_indices integer[] NOT NULL DEFAULT '{}',
    source_content_types text[] NOT NULL DEFAULT '{}',
    embedding vector(4096),
    embedding_model text,
    embedding_created_at timestamptz,
    raw_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS document_chunks_source_order_uidx
    ON document_chunks (doc_id, source_file, chunk_index);

CREATE INDEX IF NOT EXISTS document_chunks_doc_id_idx
    ON document_chunks (doc_id);

CREATE INDEX IF NOT EXISTS document_chunks_source_file_idx
    ON document_chunks (source_file);

CREATE INDEX IF NOT EXISTS document_chunks_chunk_index_idx
    ON document_chunks (chunk_index);

CREATE INDEX IF NOT EXISTS document_chunks_pages_idx
    ON document_chunks (page_start, page_end);

CREATE INDEX IF NOT EXISTS document_chunks_section_path_gin_idx
    ON document_chunks USING gin (section_path);

CREATE INDEX IF NOT EXISTS document_chunks_source_content_types_gin_idx
    ON document_chunks USING gin (source_content_types);

CREATE INDEX IF NOT EXISTS document_chunks_raw_metadata_gin_idx
    ON document_chunks USING gin (raw_metadata);

CREATE INDEX IF NOT EXISTS document_chunks_text_fts_idx
    ON document_chunks USING gin (to_tsvector('english', text));

-- pgvector HNSW indexes support up to 2000 dimensions for vector.
-- Qwen3-Embedding-8B produces 4096-dimensional embeddings, so we index
-- the first 2000 dimensions for ANN candidate retrieval and rerank by the
-- full 4096-dimensional vector in search_pgvector_mlx.py.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_subvector_hnsw_idx
    ON document_chunks
    USING hnsw ((subvector(embedding, 1, 2000)::vector(2000)) vector_cosine_ops)
    WHERE embedding IS NOT NULL;
