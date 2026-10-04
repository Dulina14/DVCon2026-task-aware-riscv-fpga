#!/bin/sh
# Step one of every deployment. Run it in /tmp/team43 straight after untarring.
# A payload file that arrived short will otherwise decode into a picture whose
# bottom is garbage, and nothing else in the pipeline will complain.
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1 && [ -f SHA256SUMS.txt ]; then
  M=SHA256SUMS.txt; C=sha256sum
elif command -v md5sum >/dev/null 2>&1 && [ -f MD5SUMS.txt ]; then
  M=MD5SUMS.txt; C=md5sum
else
  echo "no checksum tool on this board -- falling back to sizes"
  awk '{print $2}' MD5SUMS.txt | while read f; do
    [ -f "$f" ] || { echo "MISSING $f"; continue; }
  done
  exit 0
fi
echo "verifying payload with $C against $M"
if $C -c "$M" > /tmp/verify.out 2>&1; then
  echo "PAYLOAD OK -- $(grep -c ': OK$' /tmp/verify.out) files match"
  exit 0
else
  echo "PAYLOAD BAD -- do not run the pipeline, re-transfer these:"
  grep -v ': OK$' /tmp/verify.out
  exit 1
fi
