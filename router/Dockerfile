FROM golang:1.22-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o router ./cmd/server

FROM scratch
COPY --from=builder /build/router /router
EXPOSE 8081
ENTRYPOINT ["/router"]
