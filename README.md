# Build
docker build -t polymarket-bot .

# Run with secrets passed at runtime
docker run --env-file .env polymarket-bot