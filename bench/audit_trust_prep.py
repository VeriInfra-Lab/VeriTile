#!/usr/bin/env python3
"""Emit an audited temp copy of a standalone bench Lean file.

Given a path to a bench Lean file (a `bench/tritonbench_g/*/*.lean` port, a
`bench/examples/**/*.lean` showcase, or a `bench/tests/*.lean` smoke), print to
stdout a copy that:

  1. adds `import VeriTile.Meta.StatementAudit` to the import block, and
  2. appends Lean's `#auditModuleAxioms` and `#auditModuleSpecs` gates, plus
     explicit `#axiomsClean` commands for manifest entries.

Compiling the emitted copy with `lake env lean` therefore both re-checks the
port AND runs the trust audit — an external gate that never touches the port
files themselves. `bench/audit_trust.sh` drives this per file, in parallel.

Headline theorems are registered by the `specification` elaboration. Lean also
recognizes the legacy theorem-name suffixes. Python only reads exact manifest
names; it never parses Lean declarations to decide which proofs to audit.
"""
import sys
from pathlib import Path
import re
from audit_source import lean_code_mask

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts/kernel-manifest.tsv"

# These fixtures deliberately contain rejected specs inside #guard_msgs.
# Their own commands assert those failures; re-auditing the final fixture
# environment would reject the intentionally retained negative examples.
NEGATIVE_SPEC_FIXTURES = {
    "bench/tests/StatementAudit.lean",
    "bench/tests/ModuleSpecAudit.lean",
}

def imported_modules(source):
    """Read real imports, including multiline imports of repaired sources."""
    lines = lean_code_mask(source).splitlines()
    module = r'(?:[A-Z]\w*|[A-Za-z_]\w*\.[\w.]+)'
    for i, line in enumerate(lines):
        match = re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import\s+(.*)', line)
        if not match:
            continue
        body = match[1]
        j = i + 1
        while j < len(lines) and re.fullmatch(r'\s+' + module + r'(?:\s+' + module + r')*\s*', lines[j]):
            body += ' ' + lines[j].strip()
            j += 1
        for name in body.split():
            if not re.fullmatch(r'[A-Za-z_][\w.]*', name):
                raise ValueError('unsupported source import: ' + name)
            yield name

def manifest_names_for(rel_path, manifest=MANIFEST):
    names = []
    if not manifest.exists():
        return names
    for raw in manifest.read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        cols = raw.split("\t")
        if cols[0] == "id" or len(cols) < 5:
            continue
        if cols[1] == rel_path and cols[4] == "proven":
            names.append(cols[2])
    return names


def prepare_source(text, rel, manifest=MANIFEST):
    lines = text.splitlines()

    # Explicit manifest entries supplement the Lean-registered headline set.
    axioms_targets = list(dict.fromkeys(manifest_names_for(rel, manifest)))

    # Insert before the first import, preserving multiline import declarations.
    first_import = next((i for i, line in enumerate(lean_code_mask(text).splitlines())
                         if re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import\b', line)), 0)
    new_lines = list(lines)
    audit_import = "import VeriTile.Meta.StatementAudit"
    if audit_import not in new_lines:
        new_lines.insert(first_import, audit_import)

    footer = ["", "-- ==== external trust audit (appended by bench/audit_trust.sh) ===="]
    footer.append("#auditModuleAxioms")
    for t in axioms_targets:
        footer.append(f"#axiomsClean {t}")
    # Lean discovers definitions by elaborated result type, rather than a
    # first-line regex that mistakes `(kernel : ComputeKernel)` parameters
    # for return types. The command prints its actual coverage inventory.
    if rel in NEGATIVE_SPEC_FIXTURES:
        footer.append('#eval IO.println "Spec audit: negative-test fixture; guarded checks executed in source"')
    else:
        # Audit against the real imported source closure, including repaired
        # tasks and thin reader/IO forwarding modules. The frozen judge tree
        # supplies these files during replay; no historical catalog selects a
        # different kernel on the proof's behalf.
        root = manifest.parent.parent
        seen = set()
        def visit(source):
            for module in imported_modules(source):
                if not module.startswith('bench.') or module in seen:
                    continue
                seen.add(module)
                path = root.joinpath(*module.split('.')).with_suffix('.lean')
                if path.is_file():
                    visit(path.read_text())
        visit(text)
        if seen:
            footer.append(f"#auditModuleSpecs from [{', '.join(sorted(seen))}]")
        else:
            footer.append("#auditModuleSpecs")

    return "\n".join(new_lines + footer) + "\n"


def main():
    path = Path(sys.argv[1]).resolve()
    rel = path.relative_to(ROOT).as_posix()
    sys.stdout.write(prepare_source(path.read_text(), rel))
    sys.stderr.write(f"{rel}\taxiomsClean=Lean-environment\tspecNonCircular=Lean-environment\n")


if __name__ == "__main__":
    main()
