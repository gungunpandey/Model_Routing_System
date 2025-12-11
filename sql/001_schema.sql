CREATE TABLE IF NOT EXISTS requests (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  user_id TEXT,
  prompt TEXT NOT NULL,
  difficulty TEXT,
  difficulty_conf NUMERIC,
  routed_model TEXT NOT NULL,
  provider_request_id TEXT,
  input_tokens INT,
  output_tokens INT,
  latency_ms INT,
  cost_input_us INT,
  cost_output_us INT,
  total_cost_us INT,
  status TEXT,
  response_preview TEXT
);

CREATE TABLE IF NOT EXISTS cost_rates (
  id SERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  input_cost_per_1k_us INT NOT NULL,
  output_cost_per_1k_us INT NOT NULL,
  effective_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS model_eval (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  request_id BIGINT REFERENCES requests(id),
  human_score INT,
  pass_fail BOOLEAN,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS requests_ts_idx ON requests (ts);
CREATE INDEX IF NOT EXISTS requests_model_idx ON requests (routed_model);
CREATE INDEX IF NOT EXISTS requests_difficulty_idx ON requests (difficulty);

-- Seed example cost rates (micros per 1k tokens). Update as pricing changes.
INSERT INTO cost_rates (model_name, input_cost_per_1k_us, output_cost_per_1k_us, effective_date)
VALUES
  ('phi-3', 50, 100, CURRENT_DATE),
  ('llama-3', 150, 300, CURRENT_DATE),
  ('gpt-4o', 500, 1500, CURRENT_DATE)
ON CONFLICT DO NOTHING;

