#!/usr/bin/env bash
# Scan this repo for private infrastructure details and credentials before publishing; exit 1 on any finding.
#
#   scripts/check-public.sh             every tracked and untracked (not ignored) file, the patch series included
#   scripts/check-public.sh --history   also every blob, message and identity of every commit
#
# Your own private names (host names, user names, internal tools) belong in .check-public.local (git-ignored, one
# extended regex per line, # comments allowed) or in CHECK_PUBLIC_EXTRA='a|b'; they match case-insensitively.
# .check-public.allow (tracked) lists upstream identifiers a rule mistakes for private names, one extended regex a
# line; a finding line matching one is dropped.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
self=scripts/check-public.sh
local_list=.check-public.local
allow_list=.check-public.allow
allowed_authors=${ALLOWED_AUTHORS:-58495321+jayleaton@users.noreply.github.com}

# name|regex (grep -E, case-insensitive)
patterns=(
    'private IPv4 192.168/16|\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b'
    'private IPv4 10/8|\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b'
    'private IPv4 172.16/12|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b'
    'CGNAT IPv4|\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b'
    'home directories|/hom[e]/[a-z0-9_.-]+/|/User[s]/[A-Za-z0-9_.-]+/|(^|[^a-z0-9_.-])~/[a-z]'
    'machine host names|\bdg[x]-?[0-9]{1,2}\b|\bnod[e]-?[0-9]{2}\b|\bspar[k]-?[0-9]{2}\b'
    'gmail address|[a-z0-9._%+-]+@gmail\.com'
    'SSH key|ssh-(ed25519|rsa|dss) AAA[A]|BEGI[N] [A-Z ]*PRIVATE KEY'
    'tokens|\bh[f]_[A-Za-z0-9]{30,}|\bg[h][pousr]_[A-Za-z0-9]{30,}|githu[b]_pat_[A-Za-z0-9_]{20,}|\bs[k]-[A-Za-z0-9_-]{20,}|\bAKI[A][0-9A-Z]{16}\b|xo[x][abprs]-[A-Za-z0-9-]{10,}'
    'credential assignments|(api[_-]?key|token|secret|password)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"'[:space:]]{12,}'
    'tool co-author trailers|^co-author[e]d-by:'
)
if [[ -f "$local_list" ]]; then
    while IFS= read -r line; do
        [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue
        patterns+=("local list|$line")
    done < "$local_list"
fi
[[ -n "${CHECK_PUBLIC_EXTRA:-}" ]] && patterns+=("CHECK_PUBLIC_EXTRA|$CHECK_PUBLIC_EXTRA")

files=()
while IFS= read -r -d '' f; do
    [[ "$f" == "$self" || "$f" == "$local_list" || "$f" == "$allow_list" ]] && continue
    [[ -f "$f" ]] && files+=("$f")
done < <(git ls-files -z -co --exclude-standard)
[[ ${#files[@]} -gt 0 ]] || { echo 'no files to scan (not a git checkout?)' >&2; exit 2; }

fail=0
report() {
    fail=1
    echo "== $1"
    cut -c1-220 <<< "$2"
}
for entry in "${patterns[@]}"; do
    name=${entry%%|*}
    re=${entry#*|}
    hits=$(grep -HnIiE -- "$re" "${files[@]}" 2>/dev/null)
    [[ -n "$hits" && -f "$allow_list" ]] && hits=$(grep -vE -f <(grep -vE '^[[:space:]]*(#|$)' "$allow_list") <<< "$hits")
    [[ -z "$hits" ]] || report "$name" "$hits"
done
# e-mail addresses other than the allowed identities, the patches' noreply authors and reserved example domains
emails=$(grep -HnoIiE '[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}' "${files[@]}" 2>/dev/null |
    grep -viE ':[^:]*@(([a-z0-9-]+\.)*example\.(com|org|net)|invalid)$' |
    grep -vE ':[0-9]+\+[A-Za-z0-9-]+@users\.noreply\.github\.com$' || true)
[[ -z "$emails" ]] || report 'e-mail addresses' "$emails"
bad=$(printf '%s\n' "${files[@]}" | grep -E '\.(log|out|pem|key|safetensors|gguf|bin)$|(^|/)\.env$|id_(rsa|ed25519)' || true)
[[ -z "$bad" ]] || report 'files that should not be published' "$bad"

if [[ "${1:-}" == --history ]]; then
    mapfile -t commits < <(git rev-list --all)
    echo "== history: ${#commits[@]} commit(s)"
    ids=$(git log --all --format='%h %ae %ce' | awk -v ok=" $allowed_authors " 'index(ok, " " $2 " ") == 0 || index(ok, " " $3 " ") == 0')
    [[ -z "$ids" ]] || report "commits not authored and committed as: $allowed_authors" "$ids"
    msgs=$(git log --all --format='%h %B')
    for entry in "${patterns[@]}"; do
        name=${entry%%|*}
        re=${entry#*|}
        hits=$(git grep -IinE -e "$re" "${commits[@]}" -- . ":!$self" ":!$allow_list" 2>/dev/null)
        [[ -n "$hits" && -f "$allow_list" ]] && hits=$(grep -vE -f <(grep -vE '^[[:space:]]*(#|$)' "$allow_list") <<< "$hits")
        [[ -z "$hits" ]] || report "history: $name" "$hits"
        hits=$(grep -niE -e "$re" <<< "$msgs" || true)
        [[ -z "$hits" ]] || report "history messages: $name" "$hits"
    done
fi

if [[ $fail -ne 0 ]]; then
    echo
    echo "check-public: FINDINGS above (${#files[@]} files scanned)"
    exit 1
fi
echo "check-public: clean (${#files[@]} files scanned$([[ -f "$local_list" ]] && echo ", plus $local_list"))"
