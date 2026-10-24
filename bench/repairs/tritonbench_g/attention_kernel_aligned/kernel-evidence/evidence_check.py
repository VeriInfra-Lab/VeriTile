"""Read-only consistency checks of the recorded source/compiler/CPU evidence."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
TASK = OUT.parent
SOURCE_SHA = '365cda104f62655345ad002a7d2815fba0d3c812e20c5a569e9b0214c92c85ea'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_evidence():
    binder = json.loads((OUT / 'binder-artifacts/report.json').read_text())
    precision = json.loads((OUT / 'binder-artifacts/source-binding.json').read_text())
    applicability = json.loads((OUT / 'profile-applicability.json').read_text())
    obligations = json.loads((TASK / 'numerical-obligations.json').read_text())
    assert sha(TASK / 'attention_kernel_aligned_repaired.py') == SOURCE_SHA
    assert binder['status'] == precision['status'] == 'PASS'
    assert binder['actual_JIT_signature_binder'] and not binder['GPU_execution']
    assert len(binder['results']) == len(precision['profiles']) == len(applicability['profiles']) == 8
    assert precision['source_sha256'] == SOURCE_SHA
    assert precision['typed_source_sha256'] == sha(TASK / 'Kernels.lean')
    artifacts = []
    for result, profile, app in zip(binder['results'], precision['profiles'], applicability['profiles']):
        assert result['source_sha256'] == SOURCE_SHA
        assert result['status'] == 'compiled' and result['actual_JIT_signature_binder']
        assert result['name'] == profile['profile'] == app['name']
        assert all(profile['checks'].values())
        assert sha(profile['ttir']) == profile['sha256']
        assert set(result['artifacts']) == {'ttir', 'ttgir', 'llir', 'ptx'}
        for artifact in result['artifacts'].values():
            assert sha(artifact['path']) == artifact['sha256']
            artifacts.append(artifact)
        assert result['metadata']['shared'] == app['shared_bytes']
        assert app['within_sm80_per_block_shared_limit'] == (app['shared_bytes'] <= 166912)
        assert not app['runtime_success_claim']
    assert len(artifacts) == 32
    assert applicability['maximum_shared_bytes_per_block'] == 166912
    assert all(p['within_sm80_per_block_shared_limit'] for p in applicability['profiles'])
    host = (OUT / 'host-interpreter-checks.log').read_text()
    assert 'host controls 1042 admissions 43 rejections' in host
    assert 'complete FP16 interpreter grids and byte frames: 8' in host
    assert 'original output-head counterexample reproduced and repaired' in host
    assert (OUT / 'precision-check.log').read_text().strip() == 'PASS 8 profiles'
    assert obligations['deferred_numerical_equalities'] == obligations['deferred_numerical_experiments'] == []
    return {'binder': binder, 'precision': precision, 'applicability': applicability,
            'obligations': obligations, 'artifacts': artifacts}
