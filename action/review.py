#!/usr/bin/env python3
"""Sticky PR review via Mistral. Stdlib only. Fails open (warns, never breaks CI)."""
import json
import os
import subprocess
import sys
import urllib.request

MARKER = "<!-- mini-coderabbit -->"
MODEL = os.environ.get("MODEL", "mistral-small-latest")
MAX_DIFF = int(os.environ.get("MAX_DIFF_CHARS", "12000"))
FAIL_ON_ERROR = os.environ.get("FAIL_ON_ERROR", "false").lower() == "true"

SYSTEM = (
    "You review pull-request diffs. Be concise, concrete, kind. "
    "Output markdown with exactly these sections:\n"
    "## Summary (2-3 sentences)\n"
    "## Issues (bullets with file:line where possible, severity High/Med/Low, "
    "or 'No blocking issues found')\n"
    "## Suggestions (small, optional improvements, max 5)"
)

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()

def get_diff():
    base = os.environ.get("GITHUB_BASE_REF", "")
    if base:
        sh(f"git fetch origin {base} --depth=50 2>/dev/null || true")
        diff = sh(f"git diff origin/{base}...HEAD -- . ':(exclude)package-lock.json' ':(exclude)pnpm-lock.yaml' ':(exclude)poetry.lock'")
    else:
        diff = sh("git diff HEAD~1...HEAD")
    return diff[:MAX_DIFF]

def mistral(diff):
    key = os.environ["MISTRAL_API_KEY"]
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Review this diff:\n\n```diff\n{diff}\n```"},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def gh(path, method="GET", data=None):
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}{path}", data=data,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r) if r.status != 204 else None

def post_or_update(pr, body):
    full = f"{MARKER}\n🤖 **AI Review** (`{MODEL}`)\n\n{body}"
    comments = gh(f"/issues/{pr}/comments")
    mine = [c for c in comments if MARKER in c.get("body", "")]
    payload = json.dumps({"body": full}).encode()
    if mine:
        gh(f"/issues/comments/{mine[-1]['id']}", "PATCH", payload)
    else:
        gh(f"/issues/{pr}/comments", "POST", payload)

def main():
    pr = os.environ.get("PR_NUMBER") or sh("gh pr view --json number -q .number 2>/dev/null") or ""
    if not pr and (ref := os.environ.get("GITHUB_REF", "")).startswith("refs/pull/"):
        pr = ref.split("/")[2]
    diff = get_diff()
    if not pr:
        print("No PR number found; printing review to stdout.")
    if not diff.strip():
        print("Empty diff, nothing to review.")
        return
    try:
        review = mistral(diff)
    except Exception as e:
        msg = f"⚠️ AI review skipped (API error: {e})."
        print(msg, file=sys.stderr)
        if pr and os.environ.get("GITHUB_TOKEN"):
            post_or_update(pr, msg)
        if FAIL_ON_ERROR:
            sys.exit(1)
        return
    if pr and os.environ.get("GITHUB_TOKEN"):
        post_or_update(pr, review)
    print(review)

if __name__ == "__main__":
    if "MISTRAL_API_KEY" not in os.environ and "--check" not in sys.argv:
        sys.exit("Set MISTRAL_API_KEY")
    main()
