import re,hashlib
from pathlib import Path
from lean_import_scan import lean_code_mask

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def imports(text):
    """Actual project import syntax; comments, strings and quotations are blanked.

    Multiline imports are supported until the next command. Unknown import
    tokens are rejected instead of silently omitting a dependency.
    """
    code = lean_code_mask(text)
    result = []
    lines = code.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import\s+(.*)', line)
        if not m:
            continue
        body = m[1].strip()
        j = i + 1
        while j < len(lines) and re.fullmatch(r'\s+[A-Z][A-Za-z0-9_.]*(?:\s+[A-Z][A-Za-z0-9_.]*)*\s*', lines[j]):
            body += ' ' + lines[j].strip()
            j += 1
        for token in body.split():
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', token) is None:
                raise ValueError('unsupported import syntax: ' + token)
            result.append(token)
    if len(result) != len(set(result)):
        raise ValueError('duplicate imports')
    return sorted(result)
