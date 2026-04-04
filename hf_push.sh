#!/bin/bash

# Hugging Face Direct Push Script
# This uploads your backend directly to your Hugging Face Space, bypassing GitHub.

echo "🚀 Starting Hugging Face Direct Push..."

# 1. Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="$SCRIPT_DIR/.env.hf"
if [ -f "$ENV_FILE" ]; then
    echo "🔑 Loading credentials from .env.hf..."
    source "$ENV_FILE"
fi

if [ -z "$HF_USERNAME" ]; then
    read -p "Enter your Hugging Face Username (e.g. ankurpd): " HF_USERNAME
fi
if [ -z "$HF_SPACE_NAME" ]; then
    read -p "Enter your Space Name (e.g. pd-measurement-api): " HF_SPACE_NAME
fi
if [ -z "$HF_TOKEN" ]; then
    read -p "Enter your HF Access Token (Write permission): " HF_TOKEN
fi

# Save credentials if they were not already in the file
if [ ! -f "$ENV_FILE" ] && [ ! -z "$HF_TOKEN" ]; then
    echo "Saving credentials to .env.hf for next time..."
    echo "HF_USERNAME=\"$HF_USERNAME\"" > "$ENV_FILE"
    echo "HF_SPACE_NAME=\"$HF_SPACE_NAME\"" >> "$ENV_FILE"
    echo "HF_TOKEN=\"$HF_TOKEN\"" >> "$ENV_FILE"
    chmod 600 "$ENV_FILE"
fi

HF_REMOTE="https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"

# 2. Preparation
echo "📁 Preparing backend folder for push..."
cd backend

# Initialize a temporary git repo if not exists
if [ ! -d ".git" ]; then
    git init -b main
fi

# Add the Hugging Face remote
git remote remove hf 2>/dev/null
git remote add hf "$HF_REMOTE"

# 3. Deployment
echo "📦 Committing and Pushing to Hugging Face..."
git add .
git commit -m "Deployment: Direct Push to Hugging Face" 2>/dev/null || echo "No changes to commit"

# Push to Hugging Face (forced to ensure the Space matches our code)
git push hf main --force

echo ""
echo "✅ Push complete!"
echo "📍 View your Space here: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"
echo "⏳ Wait 1-2 minutes for the Docker build to finish."
echo ""
echo "🔗 Once 'Running', your API URL will be: hf.space URL"
