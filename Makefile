# porter build + run. Everything funnels through the pinned Go in Docker

# golang image that pins the toolchain (also used by the Dockerfile build stage)
GOLANG   := golang:1.26.6-alpine
IMAGE    := porter:dev

# Common mounts for the dev loop: source + persistent Go caches
MOUNTS   := -v "$(CURDIR):/app" -w /app -v gomod:/go/pkg/mod -v gocache:/root/.cache/go-build

# Pass .env through only if it exists so test/vet work before setup
DEV      := docker run --rm $(if $(wildcard .env),--env-file .env,) $(MOUNTS) $(GOLANG)
GO       := $(DEV) go

# Host arch for the native (macOS) binary
UNAME_M  := $(shell uname -m)
GOARCH   := $(if $(filter x86_64,$(UNAME_M)),amd64,arm64)

# Go sources; the native binary rebuilds when these change
SRC      := $(shell find cmd internal -name '*.go' -not -name '*_test.go') go.mod go.sum

.PHONY: test vet build server host repl run shell

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
	docker run -it --rm --init -p 127.0.0.1:8787:8787 \
		--env-file .env -e PORTER_ADDR=0.0.0.0:8787 \
		$(MOUNTS) $(GOLANG) sh -c 'go run ./cmd/porter server'

## porter-macos: native macOS binary built with the pinned Go, run on the host so the shell is the Mac's
porter-macos: $(SRC)
	$(DEV) sh -c 'CGO_ENABLED=0 GOOS=darwin GOARCH=$(GOARCH) go build -o porter-macos ./cmd/porter'

## host: run the persistent execution host agent on this machine (URL + basic
## auth from .env). The web UI's "New chat" picker then lists this host, and
## new chats can run their agent loops here. Run it from the directory you
## want provisioned chats to work in; a chat can request a different one when
## it's created.
host: porter-macos
	$(if $(wildcard .env),set -a; . ./.env; set +a;) ./porter-macos host

## repl: interactive REPL client against the server (URL + basic auth from .env)
repl: porter-macos
	$(if $(wildcard .env),set -a; . ./.env; set +a;) ./porter-macos

## run: one-shot client against the server. Usage: make run PROMPT="say hi"
run: porter-macos
	$(if $(wildcard .env),set -a; . ./.env; set +a;) ./porter-macos "$(PROMPT)"

## shell: interactive shell in the dev container
shell: build
	$(DEV) sh
