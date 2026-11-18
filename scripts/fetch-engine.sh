#!/usr/bin/env bash
# Fetch one backend's upstream TensorFold pin, apply its patch series 0001..LAST in order and require the recorded tree.
#
#   scripts/fetch-engine.sh cuda|metal DEST [SOURCE]
#
# SOURCE is an optional local TensorFold checkout or bundle that holds the pin; without it the pin is fetched from
# config/<backend>/tensorfold.repo. DEST must not exist.
set -euo pipefail
backend=${1:?usage: scripts/fetch-engine.sh cuda|metal DEST [SOURCE]}
dest=${2:?usage: scripts/fetch-engine.sh cuda|metal DEST [SOURCE]}
source=${3:-}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$backend" in
    cuda | metal) ;;
    *) echo 'backend must be cuda or metal' >&2; exit 2 ;;
esac
config=$root/config/$backend
read -r repo < "$config/tensorfold.repo"
read -r rev < "$config/tensorfold.rev"
read -r last < "$config/tensorfold.patches"
read -r tree < "$config/tensorfold.tree"
[[ "$repo" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]] || { echo "invalid $config/tensorfold.repo" >&2; exit 2; }
[[ "$rev" =~ ^[0-9a-f]{40}$ && "$tree" =~ ^[0-9a-f]{40}$ && "$last" =~ ^[0-9]{4}$ ]] || { echo "invalid pin in $config" >&2; exit 2; }
[[ ! -e "$dest" ]] || { echo "$dest already exists" >&2; exit 2; }

# The series must be exactly 0001..LAST, one file a number.
series=()
n=0
for p in "$root/patches/$backend"/[0-9][0-9][0-9][0-9]-*.patch; do
    [[ -f "$p" ]] || break
    name=${p##*/}
    [[ "${name:0:4}" > "$last" ]] && break
    n=$((n + 1))
    printf -v number '%04d' "$n"
    [[ "${name:0:4}" == "$number" ]] || { echo "patches/$backend/$number missing or not unique" >&2; exit 2; }
    series+=("$p")
done
printf -v number '%04d' "$n"
[[ "$number" == "$last" ]] || { echo "patches/$backend/$last missing" >&2; exit 2; }

git init -q "$dest"
git -C "$dest" remote add origin "$repo"
if [[ -n "$source" ]]; then
    git -C "$dest" fetch -q --no-tags "$source" "$rev"
else
    git -C "$dest" fetch -q --no-tags --depth=1 origin "$rev"
fi
git -C "$dest" checkout -q --detach "$rev"
# Fixed committer and dates, so the same series gives the same commits on any machine; authorship is the patches'.
git -C "$dest" -c user.name='TensorFold recipe' -c user.email=recipe@invalid -c core.autocrlf=false \
    am --quiet --committer-date-is-author-date --whitespace=nowarn "${series[@]}"
got=$(git -C "$dest" rev-parse 'HEAD^{tree}')
[[ "$got" == "$tree" ]] || { echo "patched tree $got differs from config/$backend/tensorfold.tree ($tree)" >&2; exit 2; }
echo "$backend engine in $dest: TensorFold ${rev:0:9} + patches 0001-$last, tree $tree"
