#!/bin/bash
set -euo pipefail

# Pin the git identity for this repo to the project owner so commits are
# never attributed to the Claude Code agent.
git config user.name "SoRaaN"
git config user.email "soraan.mahmoudi@gmail.com"
