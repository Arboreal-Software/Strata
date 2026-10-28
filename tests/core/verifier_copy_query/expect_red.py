"""Accept only the precise old-method completion assertion, never a crash/timeout."""
import subprocess
import sys

try:
    result = subprocess.run([sys.argv[1], "--copy-error"], capture_output=True,
                            text=True, timeout=20)
except subprocess.TimeoutExpired:
    raise SystemExit("old-method timeout is not an expected red")
text = result.stdout + result.stderr
print(text, end="")
expected = "FAIL: copy stream error must not report batch completion"
if result.returncode != 1 or expected not in text.splitlines():
    raise SystemExit("old method did not reach the exact copy-query assertion")
if any(marker in text for marker in ("AddressSanitizer", "runtime error:", "UndefinedBehaviorSanitizer")):
    raise SystemExit("sanitizer fault is not an expected red")
