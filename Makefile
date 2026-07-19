# ── Chess ML Project ──────────────────────────────────────────────────────────
# Run any command with: make <target>
# e.g. make sample, make test, make train

# ── Data ingestion (Layer 1) ──────────────────────────────────────────────────

# Download 50 random games and parse to Parquet (fast, for testing)
sample:
	uv run src/ingestion/ingest.py sample --n 50 --perf-type blitz

sample-large: 
	uv run src/ingestion/ingest.py sample --n 500 --perf-type blitz

# Parse a real dump — usage: make dump PATH=data/raw/lichess_db_...pgn.zst
dump:
	uv run src/ingestion/ingest.py dump $(PATH)

# ── Feature engineering (Layer 2) ────────────────────────────────────────────

# Compute all features from processed Parquet
features:
	uv run src/features/pipeline.py


# ── Stockfish labels (Layer 2.5) ──────────────────────────────────────────────

# Build eval database Parquet — one time only (~8GB download)
evals-build:
	uv run src/labels/evals.py build --evals-dir data/evals

# Build without re-downloading (if .zst already exists)
evals-build-skip-download:
	uv run src/labels/evals.py build --evals-dir data/evals --skip-download

# Generate labels by joining features with evals (seconds)
evals-label:
	uv run src/labels/evals.py label

# Scale up partitioning for 2M+ games
evals-build-large:
	uv run src/labels/evals.py build --evals-dir data/evals --partition-depth 3

# ── Training (Layer 3) ────────────────────────────────────────────────────────

# Train XGBoost regression model on centipawn loss
train:
	uv run src/training/train.py --features-dir data/features --models-dir models
# ── Full pipeline ─────────────────────────────────────────────────────────────

# Run everything end to end (uses labels-test for speed)
pipeline:
	make sample
	make features
	make labels-test
	make train

# ── Inspection ────────────────────────────────────────────────────────────────

# Inspect processed Parquet output
inspect-processed:
	uv run python -c "\
import duckdb; \
duckdb.sql(\"SELECT * FROM 'data/processed/*.parquet' LIMIT 5\").show(); \
duckdb.sql(\"SELECT count(*) as moves, count(distinct game_id) as games, round(avg(white_elo),0) as avg_elo FROM 'data/processed/*.parquet'\").show()"

# Inspect feature Parquet output
inspect-features:
	uv run python -c "\
import duckdb; \
duckdb.sql(\"SELECT * FROM 'data/features/features.parquet' LIMIT 5\").show()"

# Inspect labels Parquet output
inspect-labels:
	uv run python -c "\
import duckdb; \
duckdb.sql(\"SELECT move_quality, count(*) as count FROM 'data/labels/labels.parquet' GROUP BY move_quality ORDER BY count DESC\").show()"

# ── Tests ─────────────────────────────────────────────────────────────────────

# Run all tests
test:
	uv run pytest tests/ -v

# Run tests, stop at first failure
test-fast:
	uv run pytest tests/ -x -v

# ── Cleanup ───────────────────────────────────────────────────────────────────

# Delete processed Parquet files (keep raw downloads)
clean:
	rm -f data/processed/*.parquet
	rm -f data/features/*.parquet
	rm -f data/labels/*.parquet
	@echo "Parquet files deleted."

# Delete everything including raw downloads
clean-all:
	rm -f data/processed/*.parquet
	rm -f data/features/*.parquet
	rm -f data/labels/*.parquet
	rm -f data/raw/*.pgn data/raw/*.pgn.zst
	@echo "All data files deleted."

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Data ingestion:"
	@echo "    make sample              Download 50 games and parse to Parquet"
	@echo "    make dump PATH=...       Parse a full .pgn.zst dump"
	@echo ""
	@echo "  Feature engineering:"
	@echo "    make features            Compute features from processed Parquet"
	@echo ""
	@echo "  Labels:"
	@echo "    make labels-test         Run Stockfish on 500 rows (verify setup)"
	@echo "    make labels              Run Stockfish on full dataset"
	@echo ""
	@echo "  Training:"
	@echo "    make train               Train XGBoost regression model"
	@echo ""
	@echo "  Full pipeline:"
	@echo "    make pipeline            sample + features + labels-test + train"
	@echo ""
	@echo "  Inspection:"
	@echo "    make inspect-processed   Show processed Parquet stats"
	@echo "    make inspect-features    Show feature Parquet sample"
	@echo "    make inspect-labels      Show label distribution"
	@echo ""
	@echo "  Tests:"
	@echo "    make test                Run all tests"
	@echo "    make test-fast           Run tests, stop at first failure"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean               Delete all Parquet files"
	@echo "    make clean-all           Delete all data files"
	@echo ""

.PHONY: sample dump features labels-test labels train pipeline \
        inspect-processed inspect-features inspect-labels \
        test test-fast clean clean-all help