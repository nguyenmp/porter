# porter build + run. Everything funnels through the pinned Go in Docker

# golang image that pins the toolchain (also used by the Dockerfile build stage)
GOLANG   := golang:1.26.6-alpine
IMAGE    := porter:dev

# Common mounts for the dev loop: source + persistent Go caches
MOUNTS   := -v "$(CURDIR):/app" -w /app -v gomod:/go/pkg/mod -v gocache:/root/.cache/go-build

# Pass .env through only if it exists so test/vet work before setup
DEV      := docker run --rm $(if $(wildcard .env),--env-file .env,) $(MOUNTS) $(GOLANG)
GO       := $(DEV) go

# Bridge network so the REPL client container can reach the server container.
NET      := porter-dev
SERVER   := porter-server

.PHONY: test vet build run server repl shell

## test: run the Go test suite in the pinned dev container
test:
	$(GO) test ./...

## vet: run go vet in the pinned dev container
vet:
	$(GO) vet ./...

## build: build the standalone runnable image (porter:dev)
build:
	docker build -t $(IMAGE) .

## server: run the server (owns LLM + tools) in the dev container, attached so Ctrl-C stops it
server:
	docker network inspect $(NET) >/dev/null 2>&1 || docker network create $(NET)
	docker run -it --rm --init --name $(SERVER) --network $(NET) \
		--env-file .env -e PORTER_ADDR=0.0.0.0:8787 \
		$(MOUNTS) $(GOLANG) sh -c 'go run ./cmd/porter server'

## repl: interactive REPL client against the running server
repl:
	docker run -it --rm --network $(NET) --env-file .env \
		-e PORTER_SERVER_URL=http://$(SERVER):8787 \
		$(MOUNTS) $(GOLANG) sh -c 'go run ./cmd/porter'

## run: one-shot client against the running server. Usage: make run PROMPT="say hi"
run: build
	docker run --rm --network $(NET) --env-file .env \
		-e PORTER_SERVER_URL=http://$(SERVER):8787 $(IMAGE) $(PROMPT)

## shell: interactive shell in the dev container
shell: build
	$(DEV) sh