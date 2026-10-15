"""Discover legacy ports and split source/specification/proof directories.

This module selects files, never certifies a proof. Missing entry files and
ambiguous legacy sources are errors rather than silently skipped targets.
"""
from pathlib import Path
import argparse
import re

# A typed value contract may supplement an existing source entry. Helpers stay
# under Proofs/; accepting this name does not relax ambiguous-source checks.
SUPPLEMENTAL_SPEC_NAMES = ('FPCorrect.lean',)


def source_file(directory: Path) -> Path | None:
    if (directory / 'Kernels.lean').is_file():
        for name in ('Correct.lean', 'FPEquiv.lean'):
            if not (directory / name).is_file():
                raise ValueError(f'Missing split-port entry: {directory / name}')
        extra = set(directory.glob('*.lean')) - {
            directory / name for name in
            ('Kernels.lean', 'Correct.lean', 'FPEquiv.lean', *SUPPLEMENTAL_SPEC_NAMES)}
        if extra:
            raise ValueError(f'Unexpected split-port entries: {sorted(extra)}')
        return directory / 'Kernels.lean'
    supplemental = [directory / name for name in SUPPLEMENTAL_SPEC_NAMES
                    if (directory / name).is_file()]
    sources = sorted(p for p in directory.glob('*.lean')
                     if p.name not in SUPPLEMENTAL_SPEC_NAMES)
    if supplemental and not sources:
        raise ValueError(f'Missing source for supplemental contract: {directory}')
    if len(sources) > 1:
        raise ValueError(f'Ambiguous legacy port: {directory}')
    return sources[0] if sources else None


def source_files(root: Path) -> list[Path]:
    return [source for directory in sorted(root.iterdir()) if directory.is_dir()
            if (source := source_file(directory)) is not None]


def proof_files(directory: Path) -> list[Path]:
    source = source_file(directory)
    if source is None:
        return []
    return sorted(directory.rglob('*.lean'))


def specification_files(root: Path) -> list[Path]:
    files = []
    for source in source_files(root):
        if source.name == 'Kernels.lean':
            files.extend(source.parent / name for name in ('Correct.lean', 'FPEquiv.lean'))
        else:
            files.append(source)
        files.extend(source.parent / name for name in SUPPLEMENTAL_SPEC_NAMES
                     if (source.parent / name).is_file())
    return sorted(files)


def combined_text(source: Path) -> str:
    """Source first, then entry/proof text, for existing whole-port scans."""
    return '\n'.join(p.read_text() for p in [source] +
                     [p for p in proof_files(source.parent) if p != source])


def source_import_files(source: Path) -> list[Path]:
    """Source first, then its transitive *local* imports, in import order.

    Source audits must not search arbitrary proof files: an unimported legacy
    transcription could otherwise conceal a missing statement. Library imports
    are outside this discovery scope. Missing local imports and import cycles
    are errors, as they are for Lean itself.
    """
    from bench.audit_source import lean_code_mask

    prefix = f'bench.tritonbench_g.{source.parent.name}.'
    found, active, result = set(), set(), []

    def visit(path):
        if path in active:
            raise ValueError(f'Local source import cycle: {path}')
        if path in found:
            return
        if not path.is_file():
            raise ValueError(f'Missing local source import: {path}')
        found.add(path)
        active.add(path)
        result.append(path)
        code = lean_code_mask(path.read_text())
        for match in re.finditer(r'^\s*import\s+([^\n]+)', code, re.M):
            for module in match[1].split():
                if module.startswith(prefix):
                    relative = module[len(prefix):].split('.')
                    if not all(re.fullmatch(r'\w+', part) for part in relative):
                        raise ValueError(f'Unsupported local source import: {module}')
                    visit(source.parent.joinpath(*relative).with_suffix('.lean'))
        active.remove(path)

    visit(source)
    return result


def source_text(source: Path) -> str:
    """Read the reachable source view selected by explicit audit metadata.

    Unlike ``combined_text``, this follows only imports from the source entry.
    ``Source-audit Lean: <definition-or-macro-name>`` can select a later typed
    source body instead of an earlier historical transcription. Without that
    marker, the existing first-body convention remains in force. Multiple
    imported source modules need an explicit selector when the entry itself
    has no body; choosing one silently could audit an unrelated helper.
    """
    from bench.audit_source import (
        select_lean_audit_source, source_audit_metadata, lean_code_mask,
    )

    files = source_import_files(source)
    entry = source.read_text()
    texts = [p.read_text() for p in files]
    code = lean_code_mask(entry)
    explicit = source_audit_metadata(entry, 'Lean')
    if (explicit is None and 'triton {' in code
            and code.find('triton {') < code.find('macro_rules')):
        raise ValueError(f'Ambiguous literal and macro source bodies in {source}; '
                         'declare Source-audit Lean: <definition-or-macro-name>')
    if explicit is None and 'triton {' not in code:
        bodies = [p for p, text in zip(files[1:], texts[1:])
                  if 'triton {' in lean_code_mask(text)]
        if len(bodies) > 1:
            raise ValueError(f'Ambiguous imported source bodies for {source}: {bodies}; '
                             'declare Source-audit Lean: <definition-or-macro-name>')
    selected = select_lean_audit_source('\n'.join(texts), entry)
    if 'triton {' not in lean_code_mask(selected):
        raise ValueError(f'No reachable literal Triton source body for {source}')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('source', 'sources', 'proofs', 'all-proofs', 'specs'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.kind == 'source':
        source = source_file(args.directory)
        files = [source] if source else []
    elif args.kind == 'sources':
        files = source_files(args.directory)
    elif args.kind == 'specs':
        files = specification_files(args.directory)
    elif args.kind == 'all-proofs':
        files = [p for source in source_files(args.directory)
                 for p in proof_files(source.parent)]
    else:
        files = proof_files(args.directory)
    for file in files:
        print(file)


if __name__ == '__main__':
    main()
