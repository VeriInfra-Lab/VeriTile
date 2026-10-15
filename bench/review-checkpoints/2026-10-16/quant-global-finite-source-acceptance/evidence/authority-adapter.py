"""Root implementation for the independently reviewed frozen quantize package."""
from copy import deepcopy
from pathlib import Path
from adapter_common import declarations, hash_file, load

QUANT_GLOBAL_PACKAGE = (
    'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance/'
    'evidence/producer/quant-global'
)
QUANT_GLOBAL_RECEIPT = 'e1edc3543369fd52ed4d66dda58e6ea81778ea7312437adf7ebd2ed83c5e15e6'
QUANT_GLOBAL_REVIEW = (
    'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance/'
    'evidence/root-validation/review.json'
)
QUANT_GLOBAL_REVIEW_SHA256 = '676cd2d3cb245d600dacb2c2734ef38cd0fbe0f78fd9817feb490c85437af1ef'


def _extend_quant_global_authority(root, previous):
    root = Path(root).resolve()
    result = deepcopy(previous)

    def bind(relative, expected):
        relative = Path(relative)
        path = (root / relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts
                or not path.is_relative_to(root) or not path.is_file()
                or hash_file(path) != expected):
            raise ValueError('changed reviewed quantize input: ' + str(relative))
        key = 'repo:' + relative.as_posix()
        if result['input_sha256'].get(key, expected) != expected:
            raise ValueError('conflicting reviewed quantize input: ' + str(relative))
        result['input_sha256'][key] = expected
        return path

    package = Path(QUANT_GLOBAL_PACKAGE)
    receipt = load(bind(package / 'evidence/receipt.json', QUANT_GLOBAL_RECEIPT))
    review = load(bind(QUANT_GLOBAL_REVIEW, QUANT_GLOBAL_REVIEW_SHA256))
    if (not receipt['frozen'] or receipt['task'] != 'quantize_global'
            or review['reviewer'] != 'root'
            or review['status'] != 'substantively_reviewed_for_finite_scope_integration'
            or review['producer_receipt_sha256'] != QUANT_GLOBAL_RECEIPT):
        raise ValueError('unreviewed finite quantize authority')
    mapping = receipt['checkpoint_module_source_paths']
    if mapping != {p: p + '.txt' for p in receipt['modules']}:
        raise ValueError('invalid frozen quantize source mapping')
    for relative, expected in receipt['modules'].items():
        bind(package / mapping[relative], expected)
        bind(relative, expected)
    for relative, expected in receipt['artifacts'].items():
        bind(package / relative, expected)
    dependencies = load(root / package / 'evidence/dependencies-project.json')
    for relative, entry in dependencies['project'].items():
        bind(relative, entry['sha256'])
    delta = load(root / package / 'evidence/numerical-authority-delta.json')
    if delta['task'] != 'quantize_global':
        raise ValueError('wrong task in quantize delta')
    for port in delta['provenance'].values():
        ref = port['original_source']
        bind(ref['path'], ref['sha256'])
        for relative, expected in port['lean_source_sha256'].items():
            bind(relative, expected)
        for field in ['raw_ttir', 'native_compile_receipt',
                      'host_wiring_receipt', 'torch_cuda_sources']:
            ref = port[field]
            bind(package / ref['path'], ref['sha256'])
    for table in delta['requirements']:
        ref = table['declaration']
        parsed = declarations(root / ref['file']).get(ref['declaration'])
        if (parsed is None or parsed['kind'] != 'specification'
                or parsed['statement_sha256'] != ref['statement_sha256']):
            raise ValueError('changed quantize public statement')
    old_ids = {r['id'] for r in result['rules']}
    new_ids = [r['id'] for r in delta['rules']]
    old_refs = {(r['declaration']['file'], r['declaration']['declaration'])
                for r in result['requirements']}
    new_refs = [(r['declaration']['file'], r['declaration']['declaration'])
                for r in delta['requirements']]
    if (old_ids.intersection(new_ids) or len(new_ids) != len(set(new_ids))
            or set(result['provenance']).intersection(delta['provenance'])
            or old_refs.intersection(new_refs) or len(new_refs) != len(set(new_refs))):
        raise ValueError('reviewed quantize authority collision')
    result['rules'] += deepcopy(delta['rules'])
    result['provenance'].update(deepcopy(delta['provenance']))
    result['requirements'] += deepcopy(delta['requirements'])
    result['independent_source_reviews'].append({
        'package': 'repo', 'path': QUANT_GLOBAL_REVIEW,
        'sha256': QUANT_GLOBAL_REVIEW_SHA256,
    })
    result['summary'].update(
        new_scalar_families=len(result['rules']),
        scope_bindings=len(result['provenance']),
        declaration_cases=sum(len(r['cases']) for r in result['requirements']),
        registered_declarations=len(result['requirements']),
        unvalidated_numeric_instances=len(result['provenance']),
        quant_global_finite_delta=deepcopy(delta['summary']),
    )
    return result
