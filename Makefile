# porter build + run. Everything funnels through the pinned Go in Docker

# golang image that pins the toolchain (also used by the Dockerfile build stage)
GOLANG   := golang:1.26.6-alpine
IMAGE    := porter:dev

# Common mounts for the dev loop: source + persistent Go caches
MOUNTS   := -v "$(CURDIR):/app" -w /app -v gomod:/go/pkg/mod -v gocache:/root/.cache/go-build

# Pass .env through only if it exists so test/vet work before setup
DEV      := docker run --rm $(if $(wildcard .env),--env-file .env,) $(MOUNTS) $(GOLANG)
GO       := $(DEV) go

.PHONY: test vet build run shell

## test: run the Go test suite in the pinned dev container
test:
	$(GO) test ./...

## vet: run go vet in the pinned dev container
vet:
	$(GO) vet ./...

## build: build the standalone runnable image (porter:dev)
build:
	docker build -t $(IMAGE) .

## run: run the built image. Usage: make run PROMPT="say hi"
run: build
	docker run --rm --env-file .env $(IMAGE) $(PROMPT)

## shell: interactive shell in the dev container
shell: build
	$(DEV) sh
