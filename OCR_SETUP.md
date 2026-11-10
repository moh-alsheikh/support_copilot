# OCR Setup

OCR is optional. Install it only if your knowledge base contains images or scanned PDFs.

| Source | Needs |
|---|---|
| TXT, Markdown, DOCX, CSV, JSON, JSONL | Nothing extra |
| PDFs with a real text layer | Nothing extra (`pypdf`) |
| Images (`.png`, `.jpg`, `.tif`, `.bmp`, …) | Tesseract |
| Scanned or text-less PDF pages | Tesseract **and** Poppler |

The Python packages (`pytesseract`, `pdf2image`) come from `requirements.txt`. Tesseract and Poppler
are system programs that you install separately. Without them, preparation still runs: affected
pages and images are skipped with a warning.

## Check What You Have

```bash
python prepare_corpus.py --check-dependencies
```

The check *runs* each program rather than only looking for it on `PATH`, and reports one of:

| Status | Meaning |
|---|---|
| `available` | Installed and runs |
| `missing` | Not found on `PATH` |
| `broken` | Found on `PATH` but fails to start — usually a shared library removed by a package upgrade |

The Poppler check runs `pdftoppm` (or `pdftocairo`). `pdf2image` also calls **`pdfinfo`** to count
pages, so confirm it separately after installing Poppler:

```bash
pdftoppm -v && pdfinfo -v
```

## macOS

### macOS 13 or newer (Homebrew)

```bash
brew install tesseract poppler
python prepare_corpus.py --check-dependencies
```

### macOS 12 or older (Homebrew for Tesseract, MacPorts for Poppler)

Homebrew no longer ships pre-built packages for macOS 12, so it compiles everything from source.
Current Poppler (26.x) **cannot be compiled** with the newest Command Line Tools available for
macOS 12 (Apple clang 14): the build fails on `no member named 'sort' in namespace 'std::ranges'`.
MacPorts provides a working Poppler for these systems.

1. Install Tesseract with Homebrew: `brew install tesseract`.
2. Install MacPorts using the installer for your macOS version. For macOS 12 with MacPorts 2.12.6
   (check [macports.org/install.php](https://www.macports.org/install.php) for the current version
   and the installer matching your macOS):

   ```bash
   cd ~/Downloads
   curl -LO https://github.com/macports/macports-base/releases/download/v2.12.6/MacPorts-2.12.6-12-Monterey.pkg
   sudo installer -pkg MacPorts-2.12.6-12-Monterey.pkg -target /
   ```

3. **Make MacPorts the fallback, not the default.** The installer adds
   `/opt/local/bin:/opt/local/sbin` to the *front* of `PATH` in your shell profile. Find that line
   (use `~/.zprofile` instead if your shell is zsh):

   ```bash
   grep -n "opt/local" ~/.bash_profile
   ```

   It looks like `export PATH="/opt/local/bin:/opt/local/sbin:$PATH"`. Change it so the MacPorts
   paths come last:

   ```bash
   export PATH="$PATH:/opt/local/bin:/opt/local/sbin"
   ```

   Poppler pulls in MacPorts copies of common tools (curl, openssl, python). With MacPorts last,
   Homebrew's versions keep priority and only tools Homebrew lacks — Poppler — come from MacPorts.
4. Open a new terminal (restart your editor too if you run the pipeline from its terminal), then:

   ```bash
   sudo port selfupdate
   sudo port install poppler
   which pdftoppm pdfinfo           # both under /opt/local/bin
   pdftoppm -v && pdfinfo -v        # both should exit cleanly
   python prepare_corpus.py --check-dependencies   # expect: poppler_program: available
   ```

5. Stop Homebrew from retrying a Poppler build that can't succeed: `brew pin poppler`. If another
   Homebrew formula depends on Poppler (check with `brew uses --installed poppler`), leave the
   Homebrew copy installed but **unlinked** — `brew link poppler` would put its broken binaries back
   ahead of MacPorts.
6. Re-run preparation and indexing so previously skipped pages get OCR'd. Any `OCR skipped`
   warnings should be gone:

   ```bash
   python prepare_corpus.py
   python scripts/ingest_documents.py
   ```

7. Optional housekeeping, once nothing else is running:

   ```bash
   brew cleanup   # remove old versions left behind by the upgrade attempt
   brew doctor    # warnings about the unlinked poppler, and possibly MacPorts, are expected
   ```

Running two package managers side by side is safe for day-to-day use, because they live in separate
prefixes (`/usr/local` and `/opt/local`). The remaining risk is source builds picking up the other
manager's libraries. If a later `brew upgrade` fails with confusing errors, retry with `/opt/local`
temporarily removed from `PATH`.

## Ubuntu Or Debian

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
python prepare_corpus.py --check-dependencies
```

## Windows

Install Tesseract OCR and Poppler, add both executable directories to `PATH`, restart the terminal,
and run the dependency check. Installation locations vary; follow the official project packages
approved by your organization.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Check reports `broken`; logs show `Library not loaded: …dylib` | A dependency upgrade removed a library the program was built against. Example: Homebrew Poppler built against `gpgme` 1.x aborts with `libgpgmepp.6.dylib` once `gpgme` 2.x moves that library into the separate `gpgmepp` package | Upgrade or reinstall the program so it is rebuilt against current libraries (`brew upgrade poppler`). On macOS 12 or older that build fails — use MacPorts instead |
| Check reports `missing` right after a failed `brew upgrade` | Homebrew unlinks the old version before building the new one; when the build fails, nothing is linked | Install Poppler another way (MacPorts on macOS 12), or `brew link` the old version only if it actually runs |
| Works in one terminal, `missing` in another or in your editor | That shell started before your profile changed, so it lacks the new `PATH` | Open a new terminal or restart the editor |
| `OCR failed: Unable to get page count` | `pdfinfo` is missing or broken | Run `pdfinfo -v` and fix Poppler |
| Another tool disappears after a failed `brew upgrade poppler` (e.g. `gpg: command not found`) | The upgrade updates Poppler's dependencies first. If one of them fails to link — for example `gnupg`, when `/usr/local/sbin` doesn't exist — both its old and new versions end up unlinked | `sudo mkdir -p /usr/local/sbin && sudo chown $(whoami):admin /usr/local/sbin`, then `brew link gnupg` and confirm with `gpg --version` |
| Preparation warning `Page N: OCR skipped. Poppler is installed but fails to run…` | Same as `broken` above | Same fix |

## How Fallback Works

PDF pages first use `pypdf`. OCR runs only when the extracted text is shorter than `OCR_MIN_CHARS`
and `OCR_ENABLED=true`. The OCR result **replaces** the text layer only when it is longer, so a page
that is mostly a drawing keeps its original text even when OCR succeeds. For example, the dimension
drawing on page 5 of `raspberry-pi-5-product-brief.pdf` has only a footer line, and OCR reads the
same footer. Failures are reported per page and never stop unrelated files.

To see what OCR actually contributed, after `python prepare_corpus.py`:

```bash
grep -o '"extraction_method": "[a-z_]*"' data/processed/document_records.jsonl | sort | uniq -c
```

Per-file warnings are in `evaluation/reports/corpus_preparation.json`.

## OCR Quality

OCR text contains recognition errors, and they affect retrieval and answers:

- **Characters get confused.** In `bootloader-diagnostics.png`, the digit `0` is often read as `@`
  (`C0 flags` becomes `C@ flags`), much like the `I0-360` vs `IO-360` artifact in
  `docs/known-findings.md`.
- **Layout is lost.** In `gpio-pinout-diagram.png`, the two pin columns come out interleaved, so
  labels like `GPIO 14 (TXD)` survive but their pin positions don't.

Verify critical values against the original image, and keep the `extraction_method` metadata
visible during corpus review.
