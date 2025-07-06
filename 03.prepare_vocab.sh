#!/usr/bin/env bash
# Extract a lowercase, deduplicated vocabulary list from a Kaldi-style text file.
# Usage:  ./prepare_vocab.sh  data/corpus/text   [vocab.txt]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <text-file> [output-vocab]"
  exit 1
fi

TEXT_IN=$1
VOCAB_OUT=${2:-vocab.txt}

# Cut off the utterance-IDs, split into words, lowercase, dedup, sort.
cut -d ' ' -f 2- "$TEXT_IN"       | \
tr '[:space:]' '\n'               | \
tr '[:upper:]' '[:lower:]'        | \
grep -v '^[[:space:]]*$'          | \
sort -u > "$VOCAB_OUT"

echo "✅  Vocabulary written to $VOCAB_OUT  ( $(wc -l < "$VOCAB_OUT") words )"
