"""Independent source/evidence review of finite-word ReLU refinement."""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

TMP = Path('/tmp/veritile-standardization-20261015')
ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
WORK = TMP / 'relu-integer-word-refinement-wip'
PRIOR = TMP / 'relu-original-wip'
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import checked, declarations, hash_file, load, write_json

rp = checked(WORK, 'evidence/receipt.json', 'bf37c9bb89af701ca3737ce96fbc21d9a707d8a3d7c41e1a47d11a2e73c31276')
sp = checked(WORK, 'evidence/scope-review.json', 'da4d9317436b3f73bfc2aac6e1c0a795501b0af18904e7e065a185175ba60e57')
receipt, scope = load(rp), load(sp)
assert not receipt['whole_original_program_accepted'] and not scope['whole_original_program_accepted']
assert len(receipt['modules']) == 3
for group in ('modules', 'artifacts'):
    for p, h in receipt[group].items():
        checked(WORK, p, h)
for p, h in receipt['modules'].items():
    s = (WORK / p).read_text()
    assert s.endswith('\n') and not re.search(r'[ \t]+$', s, re.M)
checked(PRIOR, 'evidence/receipt.json', receipt['prior_frozen_receipt_sha256'])
for p, h in receipt['prior_modules_unchanged'].items():
    checked(PRIOR, p, h)
    checked(ROOT, p, h)
before, after = [load(WORK / ('evidence/final-dependencies-' + n + '.json')) for n in ('before','after')]
assert before == after and len(before['files']) == 177
roots = {'repository': ROOT, 'frozen_relu_package': PRIOR, 'package': WORK}
for p, row in before['files'].items():
    checked(roots[row['source_kind']], p, row['sha256'])
results = load(WORK / 'evidence/verification-results.json')
assert len(results) == 3
axioms = noncircular = 0
for r in results:
    assert r['exit_code'] == 0
    assert r['source_sha256_before'] == r['source_sha256_after'] == receipt['modules'][r['module']]
    log = checked(WORK, r['log'], r['log_sha256']).read_text()
    assert not re.search(r'\b(error|warning|sorryAx|DISALLOWED)\b', log)
    axioms += log.count('axiom footprint')
    noncircular += log.count('non-circular ✓')
assert (axioms, noncircular) == (23, 2)
claim = receipt['public_contract']
decl = declarations(WORK / claim['file'])[claim['parser_name']]
assert decl['kind'] == 'specification' and decl['statement_sha256'] == claim['statement_sha256']
audit = PRIOR / 'evidence/dtype-audit'
audit_receipt = load(audit / 'receipt.json')
for p, h in audit_receipt['artifacts'].items():
    checked(audit, p, h)
original = checked(ROOT, 'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py', scope['source_sha256'])
extracted = checked(audit, 'extracted_relu_kernel.py', audit_receipt['extracted_kernel_sha256'])
def kernel(path):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='relu_kernel')
assert ast.dump(kernel(original)) == ast.dump(kernel(extracted))
profiles = load(audit / 'frontend-profiles.json')
success = [r for r in profiles if r['status']=='frontend_generated']
assert len(profiles) == 17 and len(success) == 13
for r in success:
    checked(audit, r['ttir'], r['ttir_sha256'])
route_review = load(WORK / 'evidence/route-review.json')
route_by = {r['input']:r for r in route_review['finite_word_routes']}
names = {'bool', *('int'+str(w) for w in (8,16,32,64)), *('uint'+str(w) for w in (8,16,32,64))}
assert set(route_by) == names
graphs = []
for row in success:
    name = row['input']
    if name not in names:
        continue
    assert route_by[name]['ttir_sha256'] == row['ttir_sha256']
    ir = (audit / row['ttir']).read_text()
    definitions = dict(re.findall(r'^\s*(%\w+) = (.*)$',ir,re.M))
    stores = re.findall(r'tt.store (%\w+), (%\w+), (%\w+) : tensor<1024x!tt.ptr<f32>>',ir)
    assert len(stores)==1 and ir.count('tt.load ')==1 and ir.count('scf.if ')==1
    ptr, value, mask = stores[0]
    masks, offsets = {mask}, set()
    def pointer(v):
        if v in ('%x_ptr','%out_ptr'):
            return v
        rhs = definitions[v]
        op = rhs.split()[0]
        args = re.findall(r'%\w+',rhs)
        assert op in ('tt.splat','tt.addptr','tt.bitcast'),rhs
        if op=='tt.addptr':
            offsets.add(args[1])
        return pointer(args[0])
    assert pointer(ptr)=='%out_ptr'
    def term(v):
        rhs = definitions[v]
        op = rhs.split()[0]
        args = re.findall(r'%\w+',rhs)
        if op=='tt.load':
            assert len(args)==2 and pointer(args[0])=='%x_ptr'
            masks.add(args[1])
            return ('load',int(re.search(r'tt.ptr<i(\d+)>',rhs).group(1)))
        if op=='arith.constant':
            match = re.search(r'(?:dense<)?(0(?:\.0+e\+00)?)(?:>)? : (?:tensor<1024x)?(i\d+|f32)',rhs)
            assert match,rhs
            return ('zero',match.group(2))
        if op in ('tt.splat','arith.bitcast'):
            node = term(args[0])
            assert node[0]=='zero',rhs
            return node
        if op in ('arith.extsi','arith.extui'):
            widths = re.findall(r'(?:tensor<1024x)?i(\d+)>?',rhs)
            assert len(widths)==2,rhs
            node = term(args[0])
            if node[0]=='zero':
                return ('zero','i'+widths[1])
            return (op.split('.')[1],int(widths[0]),int(widths[1]),node)
        if op in ('arith.sitofp','arith.uitofp'):
            width = int(re.search(r'tensor<1024xi(\d+)> to tensor<1024xf32>',rhs).group(1))
            return (op.split('.')[1],width,'f32',term(args[0]))
        if op=='arith.cmpi':
            kind = rhs.split()[1].rstrip(',')
            assert kind in ('ne','sge','uge'),rhs
            return (kind,term(args[0]),term(args[1]))
        if op=='arith.select':
            assert 'tensor<1024xf32>' in rhs
            return ('select',*(term(a) for a in args))
        raise AssertionError(rhs)
    width = 1 if name=='bool' else int(re.search(r'\d+',name).group())
    signed = name.startswith('int')
    loaded = ('load',8 if name=='bool' else width)
    source = ('ne',loaded,('zero','i8')) if name=='bool' else loaded
    compare_input = ('extsi' if signed else 'extui',width,32,source) if width<32 else source
    compare_kind = 'sge' if signed or width<32 else 'uge'
    expected = ('select',(compare_kind,compare_input,('zero','i'+str(max(32,width)))),
        ('sitofp' if signed else 'uitofp',width,'f32',source),('zero','f32'))
    actual = term(value)
    assert actual==expected,(name,actual,expected)
    assert len(masks)==len(offsets)==1
    gate = re.search(r'scf.if (%\w+) \{',ir).group(1)
    condition = definitions[gate]
    assert condition.startswith('arith.cmpi eq, %pid, '),condition
    zero = re.findall(r'%\w+',condition)[1]
    assert definitions[zero].startswith('arith.constant 0 : i32 ')
    assert definitions['%pid'].startswith('tt.get_program_id x : i32')
    assert ir.index('tt.load ')<ir.index('arith.select ')<ir.index('scf.if ')<ir.index('tt.store ')
    graphs.append({'input':name,'ttir_sha256':row['ttir_sha256'],'store_value_graph':actual,
        'same_logical_offset_and_mask':True,'store_guard':'pid == 0','load_and_select_before_guard':True})
assert len(graphs)==9
dest = TMP / 'relu-integer-word-independent-review.json'
assert not dest.exists()
review = {
    'schema':1,'reviewer':'root','producer':'scope_integration_review',
    'status':'accepted_for_partial_finite_word_refinement_integration',
    'proof_defects_found':[],'whole_original_program_accepted':False,
    'receipt_sha256':hash_file(rp),'scope_review_sha256':hash_file(sp),
    'reviewed_source_sha256':receipt['modules'],
    'reviewed_input_sha256':{p:r['sha256'] for p,r in before['files'].items()},
    'evidence_validation':{'actual_exit_codes':[r['exit_code'] for r in results],
        'named_axiom_checks':axioms,'noncircularity_checks':noncircular,
        'source_fingerprints_before_equal_after':177,
        'dependency_source_kinds':dict(Counter(r['source_kind'] for r in before['files'].values())),
        'prior_sources_checked_against_package_and_repository':6,
        'original_kernel_ast_identical':True,'prior_ttir_hashes_checked':13,
        'independently_traced_store_graphs':graphs,'recompiled_by_reviewer':False},
    'public_contract':claim,
    'semantic_review':[
        'Root read all three new sources and rechecked the prior typed source and all nine integer/bool TTIR store graphs. Conversion is at source word width, before fp32 select; comparison alone extends small inputs.',
        'InputWords equates the entire tagged logical input Cell to explicit finite BitVec decoding at every offset below N. It does not assume successful execution or a correct result, and does not rely on mismatched-tag defaults.',
        'Signed two-complement decoding has bounds and a source-word roundtrip. Unsigned decoding has range and roundtrip. All 256 bool bytes normalize by nonzero to unsigned i1; signed and unsigned comparisons match the observed routes.',
        'The independent initial-word memory formula preserves pid-zero-only stores and arbitrary untouched prelaunch memory. Actual full ascending typed-grid success is constructed, including model-level zero grid/block cases; native validity for those cases stays open.',
        'Logical x=y is allowed. The constructor changes active logical input cells; y!=x is needed only for the separate lemma retaining a supplied output initialization. Neither lemma proves allocator freshness or physical byte alias handling.',
        'Every actual write has an fp32 tag. Untouched output tags are retained only under the stated prelaunch tag premise. Source-byte assembly, alignment, endianness, native load validity and source-to-InputWords correspondence remain structural.',
        'Comparison and bool normalization are closed word operations. M.fromInt/M.fromNat and the negative-branch fp32 literal remain callbacks. cmp16/cmp32 are inherited parameters unused by these integer/bool graphs; this is not a closed floating interpreter.',
        'Nine finite-word profiles refine nine of the existing twelve logical profiles; they do not add nine dtype names or close fp64, fp8 error, host/harness, finite-address, compiler or native scheduling scope.'
    ],
    'metadata_decision':{'model_kind':'backend_value_contract','dtype_scope_relation':'selected_explicit_profiles_only',
        'covered_stage':scope['suggested_covered_stage'],'new_profile_names':0,
        'native_binding_status':'not_established','original_source_view':'preserve_unchanged',
        'unclosed_numeric_callbacks':claim['unclosed_numeric_callbacks']},
    'script_sha256':hash_file(Path(__file__)),
    'boundary':'Independent source/evidence and SSA route review; no new comparator/native compilation/GPU or whole-original-program acceptance.'}
write_json(dest,review)
print(json.dumps({'status':review['status'],'sha256':hash_file(dest),'modules':3,
    'dependencies':177,'store_graphs':len(graphs),'axioms':axioms,'noncircular':noncircular},indent=2))
