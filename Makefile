.PHONY: verify fix lint format-check format type-check rust-verify rust-fmt rust-clippy rust-test install test

install:
	uv python install 3.12
	uv sync --frozen --no-default-groups --group dev

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

format:
	uv run ruff format .

type-check:
	uv run ty check

rust-fmt:
	cargo fmt --all -- --check

rust-clippy:
	@if [ "$$(uname -s)" = Darwin ]; then \
		env RUSTFLAGS='-C link-arg=-undefined -C link-arg=dynamic_lookup' PYO3_PYTHON="$$(uv run which python)" cargo clippy --locked --all-targets; \
	else \
		env PYO3_PYTHON="$$(uv run which python)" cargo clippy --locked --all-targets; \
	fi

rust-test:
	@if [ "$$(uname -s)" = Darwin ]; then \
		env RUSTFLAGS='-C link-arg=-undefined -C link-arg=dynamic_lookup' PYO3_PYTHON="$$(uv run which python)" cargo test --locked; \
	else \
		env PYO3_PYTHON="$$(uv run which python)" cargo test --locked; \
	fi

rust-verify: rust-fmt rust-clippy rust-test

python-verify: lint format-check type-check

verify: python-verify rust-verify

fix:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests/ -q
