#!/usr/bin/env python3
"""Exit code 0 = this call was rate-limited and should be retried after a wait; non-zero = not rate-limited.

Only structured fields are inspected: a call that did not error is never rate-limited. Keywords are searched only in the error JSON,
never in the model's text -- in one rerun the judge's answer contained "429" and the old plain grep misclassified it as rate-limited.
"""
import json, re, sys

try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if not d.get("is_error") and d.get("subtype") in (None, "success"):
    sys.exit(1)
blob = json.dumps(d, ensure_ascii=False)[:4000].lower()
pat = r"usage limit|rate.?limit|quota|insufficient_quota|too many requests|\b429\b|overloaded"
sys.exit(0 if re.search(pat, blob) else 1)
