#!/bin/bash
# LLMFlow - Git Setup and Push Script

# Navigate to project directory
cd /Users/marcel/WorkSpace/llmflow

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: LLMFlow framework implementation

- Core framework foundation with base classes
- Data and Service atoms implementation
- Complete queue system (protocol, manager, client, server)
- Molecules layer (authentication, validation, optimization)
- Conductor management system (in progress)
- Project structure and configuration files"

# Add remote origin (replace with your actual GitHub URL)
git remote add origin git@github.com:unixsysdev/llmflow.git

# Push to main branch
git branch -M main
git push -u origin main

echo "✅ LLMFlow pushed to GitHub successfully!"
