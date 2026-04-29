FROM golang:1.22-bookworm AS build

WORKDIR /src

# Download ONNX Runtime shared library
ARG ORT_VERSION=1.20.1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    -o /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /tmp && \
    cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so* /usr/local/lib/ && \
    rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-x64-${ORT_VERSION}

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 \
    CGO_LDFLAGS="-L/usr/local/lib" \
    LD_LIBRARY_PATH=/usr/local/lib \
    go build -o /app/server ./cmd/server

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app

COPY --from=build /app/server ./server

RUN useradd -r -u 1001 appuser
USER appuser

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so

EXPOSE 8080

CMD ["./server"]
