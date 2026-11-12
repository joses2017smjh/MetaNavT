# node:20-alpine = 20.20.2-alpine (Alpine 3.23) on 2026-11-11, pinned by digest like every image the root
# Dockerfile, docker-compose.yml and CI pull (refresh: doc/engineering-reference.md, Pinned base images).
# No Compose file or CI job builds this image.
FROM node:20-alpine@sha256:fb4cd12c85ee03686f6af5362a0b0d56d50c58a04632e6c0fb8363f609372293 as build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.* ./
RUN npm install

# Build the application
COPY . .
RUN npm run build

# ====================================
FROM build as release

CMD ["npm", "run", "start"]