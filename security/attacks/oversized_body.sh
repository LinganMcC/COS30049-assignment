#!/usr/bin/env bash
# Size-limit and validation checks with curl. Every response should be the
# shared JSON error format with the status shown in each heading.
# Usage (back end running):  bash security/attacks/oversized_body.sh
set -u
API=${API:-http://localhost:8000}
post() { curl -s -w '  -> HTTP %{http_code}\n' -H 'Content-Type: application/json' "$@" "$API/api/analyse"; }

echo '1. 1 MB body with Content-Length (expect 413, rejected before reading)'
python3 -c "print('{\"text\":\"' + 'a ' * 500000 + '\"}')" | post --data-binary @-

echo '2. 1 MB body, chunked - no Content-Length to trust (expect 413)'
python3 -c "print('{\"text\":\"' + 'a ' * 500000 + '\"}')" | post -H 'Transfer-Encoding: chunked' --data-binary @-

echo '3. Under the byte limit but 1,200 words (expect 422)'
python3 -c "print('{\"text\":\"' + 'a ' * 1200 + '\"}')" | post --data-binary @-

echo '4. Over 8,000 characters (expect 422)'
python3 -c "print('{\"text\":\"' + 'abcdefghij' * 900 + '\"}')" | post --data-binary @-

echo '5. Unexpected extra field (expect 422)'
post -d '{"text":"A perfectly normal sentence with enough words.","admin":true}'

echo '6. Not JSON (expect 422)'
post -d 'text=hello'
