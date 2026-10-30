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

# Lean's new `module` files cannot import the older VeriTile module format.
# Core-only mathematical helpers use the same standard-axiom boundary and
# reject kernel/specification declarations; comparator still replays every
# theorem, including private recursive equations, from the original source.
CORE_MODULE_AUDIT = r'''
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for entry in env.header.modules do
    unless #[`Init, `Std, `Lean, `Lake, `Mathlib, `Batteries, `Aesop, `Qq,
             `ProofWidgets, `ImportGraph, `LeanSearchClient, `Plausible, `Cli].any
        (·.isPrefixOf entry.module) do
      throwError "Core-only audit cannot import project module {entry.module}"
  if env.contains `VeriTile.Triton.ComputeKernel then
    throwError "Core-only audit cannot contain kernel declarations"
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  -- Exported module interfaces intentionally hide imported proof bodies as
  -- opaque axioms. Load the complete trusted imports before traversing them.
  let imported ← liftIO <| importModules env.header.imports {} (level := .private)
  let kernel := env.toKernelEnv
  let importedKernel := imported.toKernelEnv
  let mut pending : Array Name := #[]
  let mut checked := 0
  for (name, info) in kernel.constants.toList do
    if (env.getModuleIdxFor? name).isSome then continue
    let userName := ((privateToUserName? name).getD name).eraseMacroScopes
    if info.isDefinition && userName.getString!.endsWith "Spec" then
      throwError "Core-only audit cannot contain specification declarations: {name}"
    if let .axiomInfo _ := info then
      throwError "Core-only audit cannot declare axioms: {name}"
    if let .thmInfo _ := info then
      pending := pending.push name
      checked := checked + 1
  let mut seen : NameSet := {}
  while !pending.isEmpty do
    let name := pending.back!
    pending := pending.pop
    if seen.contains name then continue
    seen := seen.insert name
    let info := if (env.getModuleIdxFor? name).isSome then importedKernel.find? name
      else kernel.find? name <|> importedKernel.find? name
    let some info := info | throwError "Missing proof dependency: {name}"
    if let .axiomInfo _ := info then
      unless allowed.contains name do throwError "DISALLOWED axiom: {name}"
    pending := pending ++ info.type.getUsedConstants
    if let some value := info.value? then pending := pending ++ value.getUsedConstants
    if let .inductInfo value := info then pending := pending ++ value.ctors.toArray
  logInfo m!"Axiom audit: core module theorems={checked}, standard base only"
  logInfo "Spec audit: kernels=0, independentSpecs=0; core-only module checked"
'''

def imported_modules(source):
    """Read real imports, including multiline imports of repaired sources."""
    lines = lean_code_mask(source).splitlines()
    module = r'(?:[A-Z]\w*|[A-Za-z_]\w*\.[\w.]+)'
    for i, line in enumerate(lines):
        match = re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import(?:\s+all)?\s+(.*)', line)
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


def add_import(text, module):
    """Keep Lean's module/prelude header and multiline imports in place."""
    lines = text.splitlines()
    code = lean_code_mask(text).splitlines()
    first_import = next((i for i, line in enumerate(code)
                         if re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import\b', line)), None)
    if first_import is None:
        first_import = 0
        for i, line in enumerate(code):
            if not line.strip():
                continue
            if line.strip() in ('module', 'prelude'):
                first_import = i + 1
            else:
                break
    if not any(re.fullmatch(r'\s*(?:public\s+)?import\s+' + re.escape(module) + r'\s*', line)
               for line in code):
        lines.insert(first_import, 'import ' + module)
    return '\n'.join(lines) + '\n'


def prepare_source(text, rel, manifest=MANIFEST):
    if re.search(r'^\s*module\s*$', lean_code_mask(text), re.M):
        return add_import(text, 'Lean') + CORE_MODULE_AUDIT

    # Explicit manifest entries supplement the Lean-registered headline set.
    axioms_targets = list(dict.fromkeys(manifest_names_for(rel, manifest)))

    new_lines = add_import(text, 'VeriTile.Meta.StatementAudit').splitlines()

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
