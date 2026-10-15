"""Exact, independently reviewed additions to the frozen numerical authority."""
from copy import deepcopy
from pathlib import Path

from adapter_common import declarations, hash_file, load

RELU_FINITE_PACKAGE = (
    'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance/'
    'evidence/producer/relu'
)
RELU_FINITE_RECEIPT = '1b18f21b18b1347799969bd5a037074759ee50a7193de3139b33933e9e002e42'
RELU_FINITE_REVIEW = (
    'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance/'
    'evidence/root-validation/review.json'
)
RELU_FINITE_REVIEW_SHA256 = '30c9a722e03c76521da3daf7ec370c8151fb18dd3dbcd424ad185451bc99c909'


def _extend_relu_authority(root, previous):
    """Pin source, evidence, review and every port; never trust the queue itself."""
    root = Path(root).resolve()
    result = deepcopy(previous)

    def bind(relative, expected):
        relative = Path(relative)
        path = (root / relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts
                or not path.is_relative_to(root) or not path.is_file()
                or hash_file(path) != expected):
            raise ValueError('changed reviewed numerical extension input: ' + str(relative))
        key = 'repo:' + relative.as_posix()
        if result['input_sha256'].get(key, expected) != expected:
            raise ValueError('conflicting reviewed numerical input: ' + str(relative))
        result['input_sha256'][key] = expected
        return path

    package = Path(RELU_FINITE_PACKAGE)
    receipt = load(bind(package / 'evidence/receipt.json', RELU_FINITE_RECEIPT))
    review = load(bind(RELU_FINITE_REVIEW, RELU_FINITE_REVIEW_SHA256))
    if (not receipt['frozen']
            or review['reviewer'] != 'root'
            or review['status'] != 'substantively_reviewed_for_finite_scope_integration'
            or review['producer_receipt_sha256'] != RELU_FINITE_RECEIPT):
        raise ValueError('unreviewed finite ReLU numerical authority')
    mapping = receipt['checkpoint_module_source_paths']
    if mapping != {rel: rel + '.txt' for rel in receipt['modules']}:
        raise ValueError('invalid frozen ReLU source mapping')
    for relative, expected in receipt['modules'].items():
        bind(package / mapping[relative], expected)
        bind(relative, expected)
    for relative, expected in receipt['artifacts'].items():
        bind(package / relative, expected)
    dependencies = load(root / package / 'evidence/dependencies-before.json')
    for relative, entry in dependencies['project'].items():
        bind(relative, entry['sha256'])

    delta = load(root / package / 'evidence/numerical-authority-delta.json')
    if delta['task'] != 'relu_triton_kernel':
        raise ValueError('wrong task in reviewed ReLU numerical delta')
    for port in delta['provenance'].values():
        source = port['original_source']
        bind(source['path'], source['sha256'])
        for relative, expected in port['lean_source_sha256'].items():
            bind(relative, expected)
        for module, key in [('TypedValue', 'typed_route_source_sha256'),
                            ('FP64Source', 'fp64_route_source_sha256')]:
            bind('bench/tritonbench_g/relu_triton_kernel/Proofs/' + module + '.lean', port[key])
    for table in delta['requirements']:
        ref = table['declaration']
        parsed = declarations(root / ref['file']).get(ref['declaration'])
        if (parsed is None or parsed['kind'] != 'specification'
                or parsed['statement_sha256'] != ref['statement_sha256']):
            raise ValueError('changed finite ReLU public statement')
    old_ids = {r['id'] for r in result['rules']}
    new_ids = [r['id'] for r in delta['rules']]
    old_refs = {(r['declaration']['file'], r['declaration']['declaration'])
                for r in result['requirements']}
    new_refs = [(r['declaration']['file'], r['declaration']['declaration'])
                for r in delta['requirements']]
    if (old_ids.intersection(new_ids) or len(new_ids) != len(set(new_ids))
            or set(result['provenance']).intersection(delta['provenance'])
            or old_refs.intersection(new_refs) or len(new_refs) != len(set(new_refs))):
        raise ValueError('reviewed numerical extension collision')
    result['rules'] += deepcopy(delta['rules'])
    result['provenance'].update(deepcopy(delta['provenance']))
    result['requirements'] += deepcopy(delta['requirements'])
    result['independent_source_reviews'].append({
        'package': 'repo', 'path': RELU_FINITE_REVIEW, 'sha256': RELU_FINITE_REVIEW_SHA256,
    })
    result['summary'].update(
        new_scalar_families=len(result['rules']),
        scope_bindings=len(result['provenance']),
        declaration_cases=sum(len(r['cases']) for r in result['requirements']),
        registered_declarations=len(result['requirements']),
        unvalidated_numeric_instances=len(result['provenance']),
        relu_finite_delta=deepcopy(delta['summary']),
    )
    return result


VECTOR_FINITE_PACKAGE = (
    "bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance/"
    "evidence/producer/vector"
)
VECTOR_FINITE_RECEIPT = "8349ecfd8051f74130acf58fb8f8aa0b62d4760203188aa392a5f8670c0ec7d8"
VECTOR_FINITE_REVIEW = (
    "bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance/"
    "evidence/root-validation/review.json"
)
VECTOR_FINITE_REVIEW_SHA256 = "8f1b98f79b4df50ebb065bc31153218590808f576220d6fb2fc26aa3db4b4840"


def _extend_vector_authority(root, previous, package_path, receipt_sha256, review_path, review_sha256):
    root = Path(root).resolve()
    result = deepcopy(previous)

    def bind(relative, expected):
        rel = Path(relative)
        p = (root / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or (not p.is_relative_to(root)) or (not p.is_file()) or (hash_file(p) != expected):
            raise ValueError('changed reviewed vector input: ' + str(rel))
        key = 'repo:' + rel.as_posix()
        if result['input_sha256'].get(key, expected) != expected:
            raise ValueError('conflicting vector input')
        result['input_sha256'][key] = expected
        return p
    package = Path(package_path)
    receipt = load(bind(package / 'evidence/receipt.json', receipt_sha256))
    review = load(bind(review_path, review_sha256))
    if not receipt['frozen'] or review['reviewer'] != 'root' or review['status'] != 'substantively_reviewed_for_finite_scope_integration' or (review['producer_receipt_sha256'] != receipt_sha256):
        raise ValueError('unreviewed finite vector authority')
    mapping = receipt['checkpoint_module_source_paths']
    if mapping != {p: p + '.txt' for p in receipt['modules']}:
        raise ValueError('invalid frozen vector source mapping')
    for rel, h in receipt['modules'].items():
        bind(package / mapping[rel], h)
        bind(rel, h)
    for rel, h in receipt['artifacts'].items():
        bind(package / rel, h)
    deps = load(root / package / 'evidence/dependencies-before.json')
    for rel, entry in deps['project'].items():
        bind(rel, entry['sha256'])
    delta = load(root / package / 'evidence/numerical-authority-delta.json')
    if delta['task'] != 'vector_addition':
        raise ValueError('wrong task in vector delta')
    for port in delta['provenance'].values():
        for field in ['original_source', 'scalar_source']:
            ref = port[field]
            bind(ref['path'], ref['sha256'])
        for field in ['execution_source', 'host_source']:
            ref = port[field]
            bind(ref['module'], ref['sha256'])
        for rel, h in port['lean_source_sha256'].items():
            bind(rel, h)
        for field in ['raw_ttir', 'frontend_receipt', 'producer_scalar_inventory']:
            ref = port[field]
            bind(package / ref['path'], ref['sha256'])
    for table in delta['requirements']:
        ref = table['declaration']
        decl = declarations(root / ref['file']).get(ref['declaration'])
        if decl is None or decl['kind'] != 'specification' or decl['statement_sha256'] != ref['statement_sha256']:
            raise ValueError('changed vector public statement')
    oldids = {r['id'] for r in result['rules']}
    newids = [r['id'] for r in delta['rules']]
    oldrefs = {(r['declaration']['file'], r['declaration']['declaration']) for r in result['requirements']}
    newrefs = [(r['declaration']['file'], r['declaration']['declaration']) for r in delta['requirements']]
    if oldids.intersection(newids) or len(newids) != len(set(newids)) or set(result['provenance']).intersection(delta['provenance']) or oldrefs.intersection(newrefs) or (len(newrefs) != len(set(newrefs))):
        raise ValueError('reviewed vector authority collision')
    result['rules'] += deepcopy(delta['rules'])
    result['provenance'].update(deepcopy(delta['provenance']))
    result['requirements'] += deepcopy(delta['requirements'])
    result['independent_source_reviews'].append({'package': 'repo', 'path': str(review_path), 'sha256': review_sha256})
    result['summary'].update(new_scalar_families=len(result['rules']), scope_bindings=len(result['provenance']), declaration_cases=sum((len(r['cases']) for r in result['requirements'])), registered_declarations=len(result['requirements']), unvalidated_numeric_instances=len(result['provenance']), vector_finite_delta=deepcopy(delta['summary']))
    return result


def extend_reviewed_authority(root, previous):
    """Apply independent literal pins while preserving exact whole-queue equality."""
    result = _extend_relu_authority(root, previous)
    return _extend_vector_authority(
        root, result, VECTOR_FINITE_PACKAGE, VECTOR_FINITE_RECEIPT,
        VECTOR_FINITE_REVIEW, VECTOR_FINITE_REVIEW_SHA256,
    )
