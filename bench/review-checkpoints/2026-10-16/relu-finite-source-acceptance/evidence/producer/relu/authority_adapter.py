"""Proposed fail-closed adapter for a root-reviewed, frozen ReLU authority.

Integration point: derive_authority returns extend_authority(root, data,
reviewed_package_path, literal_reviewed_receipt_sha256). Do not infer approval
from this producer file. The root review must pin the exact receipt hash.
Existing validate_exact_queue equality remains unchanged.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

def extend_authority(root, previous, package_path, expected_receipt_sha256, frozen_source_mode="checkpoint_lean_txt"):
    root=Path(root).resolve()
    package=(root/package_path).resolve()
    if not package.is_relative_to(root):raise ValueError('package escapes repository')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    load=lambda p:json.loads(p.read_text())
    receipt_path=package/'evidence/receipt.json'
    if sha(receipt_path)!=expected_receipt_sha256:raise ValueError('unreviewed ReLU receipt')
    receipt=load(receipt_path)
    if not receipt['frozen']:raise ValueError('ReLU package is not frozen')
    if frozen_source_mode not in ['checkpoint_lean_txt','producer_lean']:raise ValueError('unknown frozen source mapping')
    mapping=receipt['checkpoint_module_source_paths']
    if mapping!={rel:rel+'.txt' for rel in receipt['modules']}:raise ValueError('invalid frozen source mapping')
    for rel,expected in receipt['modules'].items():
        stored=mapping[rel] if frozen_source_mode=='checkpoint_lean_txt' else rel
        if sha(package/stored)!=expected:raise ValueError('changed frozen ReLU source: '+stored)
    for rel,expected in receipt['artifacts'].items():
        if sha(package/rel)!=expected:raise ValueError('changed ReLU artifact: '+rel)
    for rel,expected in receipt['modules'].items():
        if sha(root/rel)!=expected:raise ValueError('ReLU public/helper source changed: '+rel)
    for rel,entry in load(package/'evidence/dependencies-before.json')['project'].items():
        if sha(root/rel)!=entry['sha256']:raise ValueError('ReLU imported source changed: '+rel)
    delta=load(package/'evidence/numerical-authority-delta.json')
    # Preserve exact source interpretation, in addition to artifact integrity.
    for port in delta['provenance'].values():
        src=port['original_source']
        if sha(root/src['path'])!=src['sha256']:raise ValueError('ReLU Python source changed')
        for rel,expected in port['lean_source_sha256'].items():
            if sha(root/rel)!=expected:raise ValueError('ReLU bound helper changed')
        for file,key in [('TypedValue','typed_route_source_sha256'),('FP64Source','fp64_route_source_sha256')]:
            rel='bench/tritonbench_g/relu_triton_kernel/Proofs/'+file+'.lean'
            if sha(root/rel)!=port[key]:raise ValueError('ReLU scalar route changed')
    from adapter_common import declarations
    for table in delta['requirements']:
        ref=table['declaration']
        if declarations(root/ref['file'])[ref['declaration']]['statement_sha256']!=ref['statement_sha256']:
            raise ValueError('ReLU public statement changed')
    result=deepcopy(previous)
    oldids={r['id'] for r in result['rules']}
    if oldids&{r['id'] for r in delta['rules']}:raise ValueError('ReLU numerical rule collision')
    if set(result['provenance'])&set(delta['provenance']):raise ValueError('ReLU port collision')
    oldrefs={(r['declaration']['file'],r['declaration']['declaration']) for r in result['requirements']}
    if oldrefs&{(r['declaration']['file'],r['declaration']['declaration']) for r in delta['requirements']}:
        raise ValueError('ReLU numerical declaration collision')
    result['rules']+=deepcopy(delta['rules'])
    result['provenance'].update(deepcopy(delta['provenance']))
    result['requirements']+=deepcopy(delta['requirements'])
    result['input_sha256'][str((package/'evidence/receipt.json').relative_to(root))]=expected_receipt_sha256
    result['input_sha256'][str((package/'evidence/numerical-authority-delta.json').relative_to(root))]=sha(package/'evidence/numerical-authority-delta.json')
    result['summary'].update(new_scalar_families=len(result['rules']),scope_bindings=len(result['provenance']),
      declaration_cases=sum(len(r['cases']) for r in result['requirements']),
      registered_declarations=len(result['requirements']),unvalidated_numeric_instances=len(result['provenance']))
    result['summary']['relu_finite_delta']=delta['summary']
    return result
