-- ============================================================
--  MBG Sentiment Analysis Database Initialization
--  PostgreSQL 15
-- ============================================================

-- Drop existing table if exists
DROP TABLE IF EXISTS sentiment_results CASCADE;

-- Create main sentiment results table
CREATE TABLE sentiment_results (
    id SERIAL PRIMARY KEY,
    pipeline_row BIGINT UNIQUE NOT NULL,
    url VARCHAR(512),
    text TEXT NOT NULL,
    text_clean TEXT,
    createdAt TIMESTAMP,
    retweet_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    quote_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    is_reply BOOLEAN DEFAULT FALSE,
    is_retweet BOOLEAN DEFAULT FALSE,
    is_quote BOOLEAN DEFAULT FALSE,
    sentiment_label VARCHAR(20),
    sarcasm_label INTEGER,
    sarcasm_confidence FLOAT,
    final_sentiment VARCHAR(20) NOT NULL,
    model_confidence FLOAT,
    label_source VARCHAR(100),
    review_required BOOLEAN DEFAULT TRUE,
    ingestion_ts TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes untuk query performance
CREATE INDEX idx_final_sentiment ON sentiment_results(final_sentiment);
CREATE INDEX idx_sarcasm_label ON sentiment_results(sarcasm_label);
CREATE INDEX idx_label_source ON sentiment_results(label_source);
CREATE INDEX idx_review_required ON sentiment_results(review_required);
CREATE INDEX idx_ingestion_ts ON sentiment_results(ingestion_ts);

-- Create view untuk summary statistics
CREATE VIEW sentiment_summary AS
SELECT 
    final_sentiment,
    sarcasm_label,
    COUNT(*) as count,
    COUNT(CASE WHEN review_required = true THEN 1 END) as pending_review,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sentiment_results), 2) as percentage
FROM sentiment_results
GROUP BY final_sentiment, sarcasm_label
ORDER BY count DESC;

-- Create view untuk data review
CREATE VIEW pending_review AS
SELECT 
    id,
    pipeline_row,
    url,
    text_clean,
    sentiment_label,
    sarcasm_label,
    final_sentiment,
    model_confidence,
    sarcasm_confidence,
    label_source
FROM sentiment_results
WHERE review_required = true
ORDER BY model_confidence ASC
LIMIT 100;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON sentiment_results TO mbg_user;
GRANT SELECT ON sentiment_summary TO mbg_user;
GRANT SELECT ON pending_review TO mbg_user;

-- Log initialization
SELECT 'Database initialized successfully' as status;