"""Exact, independently reviewed additions to the frozen numerical authority."""
from copy import deepcopy
from pathlib import Path
from adapter_common import declarations, hash_file, load
RELU_FINITE_PACKAGE = 'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance/evidence/producer/relu'
RELU_FINITE_RECEIPT = '1b18f21b18b1347799969bd5a037074759ee50a7193de3139b33933e9e002e42'
RELU_FINITE_REVIEW = 'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance/evidence/root-validation/review.json'
RELU_FINITE_REVIEW_SHA256 = '30c9a722e03c76521da3daf7ec370c8151fb18dd3dbcd424ad185451bc99c909'

def _extend_relu_authority(root, previous):
    """Pin source, evidence, review and every port; never trust the queue itself."""
    root = Path(root).resolve()
    result = deepcopy(previous)

    def bind(relative, expected):
        relative = Path(relative)
        path = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or (not path.is_relative_to(root)) or (not path.is_file()) or (hash_file(path) != expected):
            raise ValueError('changed reviewed numerical extension input: ' + str(relative))
        key = 'repo:' + relative.as_posix()
        if result['input_sha256'].get(key, expected) != expected:
            raise ValueError('conflicting reviewed numerical input: ' + str(relative))
        result['input_sha256'][key] = expected
        return path
    package = Path(RELU_FINITE_PACKAGE)
    receipt = load(bind(package / 'evidence/receipt.json', RELU_FINITE_RECEIPT))
    review = load(bind(RELU_FINITE_REVIEW, RELU_FINITE_REVIEW_SHA256))
    if not receipt['frozen'] or review['reviewer'] != 'root' or review['status'] != 'substantively_reviewed_for_finite_scope_integration' or (review['producer_receipt_sha256'] != RELU_FINITE_RECEIPT):
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
        for module, key in [('TypedValue', 'typed_route_source_sha256'), ('FP64Source', 'fp64_route_source_sha256')]:
            bind('bench/tritonbench_g/relu_triton_kernel/Proofs/' + module + '.lean', port[key])
    for table in delta['requirements']:
        ref = table['declaration']
        parsed = declarations(root / ref['file']).get(ref['declaration'])
        if parsed is None or parsed['kind'] != 'specification' or parsed['statement_sha256'] != ref['statement_sha256']:
            raise ValueError('changed finite ReLU public statement')
    old_ids = {r['id'] for r in result['rules']}
    new_ids = [r['id'] for r in delta['rules']]
    old_refs = {(r['declaration']['file'], r['declaration']['declaration']) for r in result['requirements']}
    new_refs = [(r['declaration']['file'], r['declaration']['declaration']) for r in delta['requirements']]
    if old_ids.intersection(new_ids) or len(new_ids) != len(set(new_ids)) or set(result['provenance']).intersection(delta['provenance']) or old_refs.intersection(new_refs) or (len(new_refs) != len(set(new_refs))):
        raise ValueError('reviewed numerical extension collision')
    result['rules'] += deepcopy(delta['rules'])
    result['provenance'].update(deepcopy(delta['provenance']))
    result['requirements'] += deepcopy(delta['requirements'])
    result['independent_source_reviews'].append({'package': 'repo', 'path': RELU_FINITE_REVIEW, 'sha256': RELU_FINITE_REVIEW_SHA256})
    result['summary'].update(new_scalar_families=len(result['rules']), scope_bindings=len(result['provenance']), declaration_cases=sum((len(r['cases']) for r in result['requirements'])), registered_declarations=len(result['requirements']), unvalidated_numeric_instances=len(result['provenance']), relu_finite_delta=deepcopy(delta['summary']))
    return result
VECTOR_FINITE_PACKAGE = 'bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance/evidence/producer/vector'
VECTOR_FINITE_RECEIPT = '8349ecfd8051f74130acf58fb8f8aa0b62d4760203188aa392a5f8670c0ec7d8'
VECTOR_FINITE_REVIEW = 'bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance/evidence/root-validation/review.json'
VECTOR_FINITE_REVIEW_SHA256 = '8f1b98f79b4df50ebb065bc31153218590808f576220d6fb2fc26aa3db4b4840'

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
QUANT_GLOBAL_PACKAGE = 'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance/evidence/producer/quant-global'
QUANT_GLOBAL_RECEIPT = 'e1edc3543369fd52ed4d66dda58e6ea81778ea7312437adf7ebd2ed83c5e15e6'
QUANT_GLOBAL_REVIEW = 'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance/evidence/root-validation/review.json'
QUANT_GLOBAL_REVIEW_SHA256 = '676cd2d3cb245d600dacb2c2734ef38cd0fbe0f78fd9817feb490c85437af1ef'

def _extend_quant_global_authority(root, previous):
    root = Path(root).resolve()
    result = deepcopy(previous)

    def bind(relative, expected):
        relative = Path(relative)
        path = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or (not path.is_relative_to(root)) or (not path.is_file()) or (hash_file(path) != expected):
            raise ValueError('changed reviewed quantize input: ' + str(relative))
        key = 'repo:' + relative.as_posix()
        if result['input_sha256'].get(key, expected) != expected:
            raise ValueError('conflicting reviewed quantize input: ' + str(relative))
        result['input_sha256'][key] = expected
        return path
    package = Path(QUANT_GLOBAL_PACKAGE)
    receipt = load(bind(package / 'evidence/receipt.json', QUANT_GLOBAL_RECEIPT))
    review = load(bind(QUANT_GLOBAL_REVIEW, QUANT_GLOBAL_REVIEW_SHA256))
    if not receipt['frozen'] or receipt['task'] != 'quantize_global' or review['reviewer'] != 'root' or (review['status'] != 'substantively_reviewed_for_finite_scope_integration') or (review['producer_receipt_sha256'] != QUANT_GLOBAL_RECEIPT):
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
        for field in ['raw_ttir', 'native_compile_receipt', 'host_wiring_receipt', 'torch_cuda_sources']:
            ref = port[field]
            bind(package / ref['path'], ref['sha256'])
    for table in delta['requirements']:
        ref = table['declaration']
        parsed = declarations(root / ref['file']).get(ref['declaration'])
        if parsed is None or parsed['kind'] != 'specification' or parsed['statement_sha256'] != ref['statement_sha256']:
            raise ValueError('changed quantize public statement')
    old_ids = {r['id'] for r in result['rules']}
    new_ids = [r['id'] for r in delta['rules']]
    old_refs = {(r['declaration']['file'], r['declaration']['declaration']) for r in result['requirements']}
    new_refs = [(r['declaration']['file'], r['declaration']['declaration']) for r in delta['requirements']]
    if old_ids.intersection(new_ids) or len(new_ids) != len(set(new_ids)) or set(result['provenance']).intersection(delta['provenance']) or old_refs.intersection(new_refs) or (len(new_refs) != len(set(new_refs))):
        raise ValueError('reviewed quantize authority collision')
    result['rules'] += deepcopy(delta['rules'])
    result['provenance'].update(deepcopy(delta['provenance']))
    result['requirements'] += deepcopy(delta['requirements'])
    result['independent_source_reviews'].append({'package': 'repo', 'path': QUANT_GLOBAL_REVIEW, 'sha256': QUANT_GLOBAL_REVIEW_SHA256})
    result['summary'].update(new_scalar_families=len(result['rules']), scope_bindings=len(result['provenance']), declaration_cases=sum((len(r['cases']) for r in result['requirements'])), registered_declarations=len(result['requirements']), unvalidated_numeric_instances=len(result['provenance']), quant_global_finite_delta=deepcopy(delta['summary']))
    return result
QUANT_TRANSPOSE_PACKAGE = 'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/evidence/producer/quant-transpose'
QUANT_TRANSPOSE_RECEIPT = '883f7e8e086acb7b20ac7a3f20ba16dfdf697b425ef123d0af7509c877bf188a'
QUANT_TRANSPOSE_REVIEW = 'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/evidence/root-validation/quant-transpose/review.json'
QUANT_TRANSPOSE_REVIEW_SHA256 = '54c7e23fb5975ec8fddd7f7bb19304616f9895607fc9f4af313a60e29aacf918'
VECTOR_NATIVE_PACKAGE = 'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/evidence/producer/vector'
VECTOR_NATIVE_RECEIPT = '8679242cdbdbe9d8972fe59c011fc89f51ba723de0b233bbe8af07ec32312cc4'
VECTOR_NATIVE_REVIEW = 'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/evidence/root-validation/vector/review.json'
VECTOR_NATIVE_REVIEW_SHA256 = '2d5529a7a7114a7e846026d44f3d1b8a7b26455294580b7e9a48326074c3f362'

def extend_reviewed_authority(root, previous):
    """Apply independent literal pins while preserving exact whole-queue equality."""
    result = _extend_relu_authority(root, previous)
    result = _extend_vector_authority(root, result, VECTOR_FINITE_PACKAGE, VECTOR_FINITE_RECEIPT, VECTOR_FINITE_REVIEW, VECTOR_FINITE_REVIEW_SHA256)
    result = _extend_quant_global_authority(root, result)
    from quant_transpose_numerical_extension import extend_authority
    result = extend_authority(root, result, QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_RECEIPT, QUANT_TRANSPOSE_REVIEW, QUANT_TRANSPOSE_REVIEW_SHA256)
    from vector_native_numerical_extension import extend_vector_native_authority
    result = extend_vector_native_authority(root, result, VECTOR_NATIVE_PACKAGE, VECTOR_NATIVE_RECEIPT, VECTOR_NATIVE_REVIEW, VECTOR_NATIVE_REVIEW_SHA256)
    from mul2_finite_numerical_extension import extend_triton_mul2_authority as _extend_next_mul2
    result = _extend_next_mul2(root, result, 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/producer/mul2', '7644e8a8dadc81452975c71b662ff1ae587bb6ceab8fd07c94e996ce0be75a4f', 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/root-validation/mul2/review.json', '90a92dc701865fe6f888119e0d9d433a4b3f7768ff46bab2fcca256e8867917f')
    from relu_actual_numerical_extension import extend_authority as _extend_next_relu
    result = _extend_next_relu(root, result, 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/producer/relu', 'e219fb011932538d411b02b85a5e624e7d617198faca6730a953ddac2db5e46e', 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/root-validation/relu/review.json', 'a42c80179e198570684d477cf04d908ea006206f51f2066dcdca3be82be52453')
    from custom_native_numerical_extension import extend_vector_custom_native_authority as _extend_next_custom
    result = _extend_next_custom(root, result, 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/producer/custom', '261d2be00ae6152fbbad9fc6c540ae96d0ff653198e9ae3ea7533878be7de8e1', 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/root-validation/custom/review.json', '3f07f50af5b4c0c590e2e42eb0247c094284ce2a793ff7dfd564f46df2104e28')
    from add_example_numerical_extension import extend_authority as _extend_next_add_example
    result = _extend_next_add_example(root, result, 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/producer/add-example', '83920c47c558019d291230261b96f0827376831ae091430ce6971e37a640d8f5', 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/root-validation/add-example/review.json', '1f86c11730ea8241c6c74c9a087ae819d0903e8be395b58a5e043cfea2fd2119')
    return result
