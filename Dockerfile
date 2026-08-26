# Builder stage: compile with the pinned Go version.
FROM golang:1.26.6-alpine AS build
WORKDIR /src
COPY . ./
# Dependencies (chi, goldmark, modernc.org/sqlite) are vendored by go mod
# download; modernc.org/sqlite is pure Go so CGO_ENABLED=0 stays static.
# CGO_ENABLED=0 forces a static binary so the alpine runtime has no libc deps.
RUN CGO_ENABLED=0 go build -o /porter ./cmd/porter

# Runtime stage: slim Alpine
FROM alpine:3.22

# CA certs for HTTPS calls to the LLM API.
RUN apk add --no-cache ca-certificates

COPY --from=build /porter /usr/local/bin/porter
ENTRYPOINT ["/usr/local/bin/porter"]
