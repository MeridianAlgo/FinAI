.PHONY: test lint format build-docker docker-run

test:
	pytest -q

lint:
	ruff check . || true

format:
	black .

build-docker:
	docker build -t meridianalgo/finai:latest .

docker-run:
	docker run --rm -it -v $(PWD)/checkpoints:/app/checkpoints -e HF_TOKEN=$$HF_TOKEN meridianalgo/finai:latest
