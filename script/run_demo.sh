#!/usr/bin/env bash
set -e
echo "🚀 Building Docker images..."
docker-compose build
echo "🌐 Starting all services..."
docker-compose up
