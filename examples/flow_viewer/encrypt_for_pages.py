"""Encrypt flow_viewer.html into a password-protected page for GitHub Pages.

    VIEWER_PASSWORD=... python examples/flow_viewer/encrypt_for_pages.py  -> examples/flow_viewer/site/index.html

GitHub Pages only serves static files, so a password screen alone would leave the
data readable in the page source. Instead the whole page is encrypted:
PBKDF2-HMAC-SHA256 (600k iterations) derives an AES-256-GCM key from the password,
and the browser decrypts with WebCrypto. The published file holds only salt, IV
and ciphertext. The password is read from the environment, never from a file.
Requires `pip install cryptography`.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

HERE = Path(__file__).parent
ITERATIONS = 600_000

GATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Account Research Runs</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root { --ground: #eef1f4; --dots: #cbd3dc; --surface: #fff; --ink: #15202b; --muted: #5a6776; --line: #d9dfe6; --accent: #1f5faf; --on-accent: #ffffff; --bad: #b8432f; }
  @media (prefers-color-scheme: dark) { :root { --ground: #0e1419; --dots: #1f2a35; --surface: #151d25; --ink: #e4eaf0; --muted: #93a1b0; --line: #26323e; --accent: #6fa8f5; --on-accent: #0e1419; --bad: #e57a64; } }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px; color: var(--ink);
    font: 15px/1.5 "IBM Plex Sans", system-ui, sans-serif; background: var(--ground);
    background-image: radial-gradient(var(--dots) 1px, transparent 1px); background-size: 18px 18px; }
  form { width: min(380px, 100%); background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 24px; display: grid; gap: 12px; }
  h1 { font: 600 22px/1.2 "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; margin: 0; }
  p { margin: 0; color: var(--muted); font-size: 13.5px; }
  label { font-size: 13px; font-weight: 500; }
  input { font: inherit; padding: 9px 11px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); }
  button { font: 500 14px/1 "IBM Plex Sans", sans-serif; padding: 11px; border: 0; border-radius: 6px; background: var(--accent); color: var(--on-accent); cursor: pointer; }
  input:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .err { color: var(--bad); min-height: 1.4em; font-size: 13px; }
</style>
</head>
<body>
<form id="gate">
  <h1>Account Research Runs</h1>
  <p>Step-by-step mock runs of the Quorum Account Research Agent. Enter the password from the submission document.</p>
  <label for="pw">Password</label>
  <input id="pw" type="password" autocomplete="current-password" required autofocus>
  <button type="submit">Open the viewer</button>
  <div class="err" id="err" role="alert"></div>
</form>
<script>
const PAYLOAD = { salt: "__SALT__", iv: "__IV__", data: "__DATA__", iterations: __ITER__ };
const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
async function decrypt(password) {
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey({ name: "PBKDF2", salt: bytes(PAYLOAD.salt), iterations: PAYLOAD.iterations, hash: "SHA-256" },
    base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: bytes(PAYLOAD.iv) }, key, bytes(PAYLOAD.data));
  return new TextDecoder().decode(plain);
}
document.getElementById("gate").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const err = document.getElementById("err");
  err.textContent = "Checking…";
  try {
    const html = await decrypt(document.getElementById("pw").value);
    document.open(); document.write(html); document.close();
  } catch {
    err.textContent = "Wrong password. Check the submission document and try again.";
  }
});
</script>
</body>
</html>
"""


def main() -> None:
    password = os.environ.get("VIEWER_PASSWORD")
    if not password:
        sys.exit("Set VIEWER_PASSWORD in the environment.")
    viewer = (HERE / "flow_viewer.html").read_text()
    html = (
        '<!doctype html>\n<html lang="en"><head>'
        + viewer.replace('<div class="app">', '</head><body><div class="app">', 1)
        + "</body></html>"
    )
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(password.encode())
    ciphertext = AESGCM(key).encrypt(iv, html.encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    page = (
        GATE.replace("__SALT__", b64(salt))
        .replace("__IV__", b64(iv))
        .replace("__DATA__", b64(ciphertext))
        .replace("__ITER__", str(ITERATIONS))
    )
    out = HERE / "site" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    (out.parent / ".nojekyll").write_text("")
    print(f"{out} ({len(page) // 1024} KB, encrypted)")


if __name__ == "__main__":
    main()
