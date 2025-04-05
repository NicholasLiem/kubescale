# Step 1. Build binary
FROM golang:1.23.0 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN GOOS=linux GOARCH=amd64 go build -o brain-controller .

# Step 2. Build a small image

FROM alpine:latest
WORKDIR /root/

COPY --from=builder /app/brain-controller .

# Copy the env
COPY --from=builder /app/.env .

CMD ["./brain-controller"]
