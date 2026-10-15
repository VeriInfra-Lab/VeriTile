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


def extend_reviewed_authority(root, previous):
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
