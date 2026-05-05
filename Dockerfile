# Stage 1: Build environment
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime environment
FROM python:3.13-slim

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the virtual environment and the application code
COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/
COPY pyproject.toml uv.lock /app/

# Set the path to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["uv", "run", "tfm-slm"]
