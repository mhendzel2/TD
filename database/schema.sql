-- Trading Dashboard Database Schema
-- PostgreSQL Database Schema for Newsletter-Enhanced Trading Dashboard

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Newsletter sources table
CREATE TABLE newsletter_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) NOT NULL,
    priority INTEGER DEFAULT 5,
    credibility_score DECIMAL(3,2) DEFAULT 0.50,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Newsletter analysis table
CREATE TABLE newsletter_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    source_id UUID REFERENCES newsletter_sources(id),
    subject TEXT,
    content TEXT,
    extracted_tickers TEXT[],
    sentiment_score DECIMAL(4,3),
    sentiment_confidence DECIMAL(4,3),
    priority_score INTEGER,
    processed_at TIMESTAMP DEFAULT NOW(),
    raw_email_data JSONB
);

-- ML predictions table
CREATE TABLE ml_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    ticker VARCHAR(10) NOT NULL,
    newsletter_analysis_id UUID REFERENCES newsletter_analysis(id),
    probability_score DECIMAL(4,3) NOT NULL,
    confidence_lower DECIMAL(4,3),
    confidence_upper DECIMAL(4,3),
    contributing_factors JSONB,
    model_version VARCHAR(50),
    prediction_timestamp TIMESTAMP DEFAULT NOW()
);

-- Portfolio positions table
CREATE TABLE portfolio_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    ticker VARCHAR(10) NOT NULL,
    position_type VARCHAR(20), -- 'STOCK', 'CALL', 'PUT'
    quantity DECIMAL(10,2),
    avg_cost DECIMAL(10,4),
    current_price DECIMAL(10,4),
    market_value DECIMAL(12,2),
    unrealized_pnl DECIMAL(12,2),
    delta DECIMAL(8,4),
    gamma DECIMAL(8,4),
    theta DECIMAL(8,4),
    vega DECIMAL(8,4),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, ticker, position_type)
);

-- Market data table for caching
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- 'options_flow', 'volume', 'gex', etc.
    data_value JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_ticker_type_timestamp (ticker, data_type, timestamp)
);

-- User alerts table
CREATE TABLE user_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    alert_type VARCHAR(50) NOT NULL, -- 'high_probability', 'portfolio_risk', etc.
    ticker VARCHAR(10),
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trading sessions table for tracking performance
CREATE TABLE trading_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    prediction_id UUID REFERENCES ml_predictions(id),
    ticker VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    quantity DECIMAL(10,2),
    pnl DECIMAL(12,2),
    trade_success BOOLEAN, -- For ML training data
    entry_timestamp TIMESTAMP,
    exit_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_newsletter_analysis_user_processed ON newsletter_analysis(user_id, processed_at DESC);
CREATE INDEX idx_ml_predictions_ticker_timestamp ON ml_predictions(ticker, prediction_timestamp DESC);
CREATE INDEX idx_ml_predictions_user_probability ON ml_predictions(user_id, probability_score DESC);
CREATE INDEX idx_portfolio_positions_user_ticker ON portfolio_positions(user_id, ticker);
CREATE INDEX idx_market_data_ticker_type_timestamp ON market_data(ticker, data_type, timestamp DESC);
CREATE INDEX idx_user_alerts_user_created ON user_alerts(user_id, created_at DESC);
CREATE INDEX idx_trading_sessions_user_ticker ON trading_sessions(user_id, ticker);

-- Insert default newsletter sources
INSERT INTO newsletter_sources (name, domain, priority, credibility_score) VALUES
('Unusual Whales', 'unusualwhales.com', 9, 0.85),
('Flow Traders', 'flowtraders.com', 8, 0.80),
('Market Chameleon', 'marketchameleon.com', 7, 0.75),
('Options Flow', 'optionsflow.com', 8, 0.82),
('Benzinga Pro', 'benzinga.com', 6, 0.70);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolio_positions_updated_at BEFORE UPDATE ON portfolio_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

