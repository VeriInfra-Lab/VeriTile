"""Read-only inspection of existing wrapper/binder and compiler artifacts; no rerun."""
from pathlib import Path
import ast
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
TASK = OUT.parent
SOURCE = TASK / 'Whole001/attention_fwd_triton2_repaired.py'
QUOTATION = TASK / 'Whole001/Kernels.lean'
EXPECTED = 'eb7192d0aaab5db92b98eba026b05eb728774263b5f3afecbd7270d9d84bc005'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
pin = lambda path: {'path': str(path), 'sha256': sha(path)}
assert sha(SOURCE) == EXPECTED
functions = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
outer_args = [a.arg for a in functions['_attn_fwd'].args.args]
inner_args = [a.arg for a in functions['_attn_fwd_inner'].args.args]
assert len(outer_args) == 29 and len(inner_args) == 18
report = json.loads((OUT / 'binder-report.json').read_text())
assert report['source_sha256'] == EXPECTED
assert report['script_sha256'] == sha(OUT / 'binder_probe.py')
assert not report['GPU_execution'] and not report['numerical_experiment']
profiles = []
for result in report['results']:
    B,H,N,D = result['shape']
    assert result['wrapper_accepted'] and result['output_identity'] and result['allocations'] == 2
    assert result['output_shape'] == result['shape'] and result['output_dtype'] == 'torch.bfloat16'
    assert result['unused_M_shape'] == [B,H,N]
    if not B*H*N:
        assert result['status'] == 'empty_skip' and result['launch_count'] == 0
        continue
    assert result['status'] == 'compiled' and result['launch_count'] == 1
    assert result['argument_names'] == outer_args
    assert result['grid'] == [(N+127)//128,B*H,1]
    assert result['launch_scalar_values'] == [H*N*96,N*96,96,1]*4+[B,H]
    assert result['launch_options'] == dict(N_CTX=N, HEAD_DIM=96, BLOCK_M=128, BLOCK_N=64, STAGE=1, num_warps=8, num_stages=2)
    for name in outer_args[:5]:
        assert result['signature'][name] == '*fp32'
    assert result['signature']['Out'] == '*bf16'
    for name,val in dict(HEAD_DIM=96,BLOCK_M=128,BLOCK_N=64,STAGE=1).items():
        assert result['signature'][name] == 'constexpr' and result['constants'][name] == str(val)
    assert set(result['signature']) == set(outer_args)
    assert set(result['attrs']) == {str((i,)) for i in range(29)}
    assert result['compiler_options']['num_warps'] == 8
    assert result['compiler_options']['num_stages'] == 2
    assert result['compiler_options']['default_dot_input_precision'] == 'tf32'
    assert result['compiler_options']['enable_fp_fusion']
    assert set(result['artifacts']) == {'ttir','ttgir','llir','ptx'}
    texts = {}
    for kind,artifact in result['artifacts'].items():
        assert pin(Path(artifact['path'])) == artifact
        texts[kind] = Path(artifact['path']).read_text()
    ttir,ptx = texts['ttir'],texts['ptx']
    patterns = {
        'QK_FP32_TF32': r'tt.dot .*inputPrecision = tf32 : tensor<128x128xf32> \* tensor<128x64xf32> -> tensor<128x64xf32>',
        'P_FP16': r'arith.truncf .*tensor<128x64xf32> to tensor<128x64xf16>',
        'V_FP16': r'arith.truncf .*tensor<64x128xf32> to tensor<64x128xf16>',
        'PV_FP16_output': r'tt.dot .*tensor<128x64xf16> \* tensor<64x128xf16> -> tensor<128x128xf16>',
        'PV_widen_FP32': r'arith.extf .*tensor<128x128xf16> to tensor<128x128xf32>',
        'final_FP32_divide': r'arith.divf .*tensor<128x128xf32>',
        'final_BF16': r'arith.truncf .*tensor<128x128xf32> to tensor<128x128xbf16>',
        'BF16_store': r'tt.store .*tensor<128x128x!tt.ptr<bf16>>',
        'base2': r'math.exp2',
    }
    checks = {name: bool(re.search(pattern,ttir)) for name,pattern in patterns.items()}
    for token in ['mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32',
                  'mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16',
                  'cvt.rn.f16x2.f32','cvt.f32.f16','cvt.rn.bf16x2.f32',
                  'ex2.approx.ftz.f32','fma.rn.f32','div.full.f32']:
        checks[token] = token in ptx
    checks['xor2_then_xor1'] = re.findall(r'shfl.sync.bfly.b32\s+%\w+, %\w+, (\d+),',ptx) == ['2','1']*4
    checks['TF32_128_static_instructions'] = ptx.count('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32') == 128
    checks['half_64_static_instructions'] = ptx.count('mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16') == 64
    assert all(checks.values()), (result['name'],checks)
    evidence_lines = {}
    for kind,text,needles in [('ttir',ttir,['tt.dot','arith.truncf','arith.extf','arith.divf','arith.mulf','math.exp2']),
                              ('ptx',ptx,['shfl.sync.bfly','fma.rn.f32','cvt.rn.bf16','div.full.f32'])]:
        evidence_lines[kind] = {needle:[i+1 for i,l in enumerate(text.splitlines()) if needle in l] for needle in needles}
    profiles.append({'profile':result['name'],'checks':checks,'artifacts':result['artifacts'],
        'shared_bytes':result['metadata']['shared'],'runtime_success_claim':False,'evidence_lines':evidence_lines})
assert len(profiles)==5
# Full helper is inlined; these substitutions are fixed by the existing launch.
source_map = [
    {'Python':'outer program IDs, integer promotion, batch/head split and all six pointers','quotation':'start_m through O_block_ptr'},
    {'Python':'m_i/l_i/acc initialization and masked Q/QS load','quotation':'same statements and FP32 dtype'},
    {'Python':'_attn_fwd_inner lo=0,hi=N_CTX,range step BLOCK_N','quotation':'entire helper loop inlined with range(0,N_CTX,BLOCK_N)'},
    {'Python':'QK, ordered q_scale then k_scale, tail -inf, online max/exp2/sum/alpha','quotation':'same loop statements and operation order'},
    {'Python':'V and P to FP16, dot out_dtype=FP16, add into FP32 acc','quotation':'same typed conversions and FP16 dot result'},
    {'Python':'m_i update and K/KS/V pointer advances','quotation':'same three pointer advances including masked/final iteration'},
    {'Python':'acc/l_i and Out.type.element_ty cast with row/feature mask','quotation':'BF16 cast fixed by actual wrapper empty_like(dtype=bfloat16)'}]
result = {'status':'PASS','scope':'Static inspection of already saved actual wrapper/default-binder profiles and compiled artifacts; manual full-source/quotation binding map. No compiler or interpreter rerun.',
    'source':pin(SOURCE),'quotation':pin(QUOTATION),'outer_arguments':outer_args,'helper_arguments':inner_args,
    'source_binding':source_map,'fixed_substitutions':{'BLOCK_M':128,'BLOCK_N':64,'BLOCK_DMODEL':128,'HEAD_ACTIVE':96,'HEAD_DIM':96,'STAGE':1},
    'profiles':profiles,'empty_profiles':3,'native_plan_boundary':'Existing fixed source/ISA model retains full trees and zero-seeded instruction chains. Saved compiler artifacts include specialization (notably N=1 dead lanes); the evidence is not a universal compiler-to-model correspondence theorem.',
    'TF32_note':'QK fixed-ISA port owns TF32 instruction interpretation. No additional conversion opcode or conversion equality is invented.',
    'GPU_execution':False,'primitive_numerical_experiment':False,'universal_compiler_conformance_claim':False}
(OUT/'source-binding.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS five saved compiler profiles, three empty wrapper profiles, full source binding map')
