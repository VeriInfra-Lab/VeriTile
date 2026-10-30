#!/usr/bin/env python3
"""Check tracked Markdown file links and synchronized reference documentation."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INLINE = re.compile(r'''\]\(\s*(<[^>]+>|(?:\\.|[^\s()]+|\([^)]*\))+)(?:\s+["'][^"']*["'])?\s*\)''')
REFERENCE = re.compile(r'^ {0,3}\[[^]]+\]:\s*(<[^>]+>|\S+)')
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')


def links(text):
    """Read inline/reference destinations, excluding fenced and inline code."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        if fence:
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= fence[1] and not marker[2].strip()):
                fence = None
            continue
        if marker:
            fence = marker[1][0], len(marker[1])
            continue
        line = re.sub(r'(`+).*?\1', '', line)
        for match in INLINE.finditer(line):
            yield number, match[1].strip('<>')
        if match := REFERENCE.match(line):
            yield number, match[1].strip('<>')


def check_links(root, documents):
    root = root.resolve()
    errors = []
    for document in documents:
        for number, href in links(document.read_text()):
            url = urlsplit(href)
            # Absolute web routes and fragments are checked on the built site.
            if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
                continue
            name = unquote(re.sub(r'\\([() ])', r'\1', url.path))
            target = (document.parent / name).resolve()
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f'{document.relative_to(root)}:{number}: missing local link {href}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links-only', action='store_true', help='skip the generated-reference freshness check')
    args = parser.parse_args()
    raw = subprocess.check_output(['git', 'ls-files', '-z', '--', '*.md', '*.mdx'], cwd=ROOT)
    documents = [ROOT / name.decode() for name in raw.split(b'\0') if name]
    if not documents:
        raise SystemExit('No tracked Markdown documents found.')
    errors = check_links(ROOT, documents)
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'{len(documents)} tracked Markdown documents: local file links checked.')
    if not args.links_only:
        return subprocess.run([sys.executable, '-B', 'site/scripts/sync-docs.py', '--check'], cwd=ROOT).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
