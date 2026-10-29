"""Source checks, exact numerical binding proposal and frozen proof receipt."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
base = root.parent
prefix = 'bench/tritonbench_g/vector_addition/'
modules = [prefix+s+'.lean' for s in ['Proofs/TypedHostDispatch','Proofs/TypedHostExecution',
           'Proofs/TypedHostWrapper','Proofs/TypedHostValues','TypedHostCorrect']]
audit = base/'vector-dtype-admission-wip/evidence/source-audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
dump = lambda p,v: p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
contracts = ['VeriTile.Bench.TritonBenchG.VectorAddition.TypedHostCorrect.'+s for s in
 ['vector_addition_typed_host_success','vector_addition_typed_host_outcome',
  'vector_addition_typed_host_allocation_error','vector_addition_typed_host_cuda_error']]

def dependencies():
    project, external = {}, {}
    queue = modules[:]
    while queue:
        rel = queue.pop()
        if rel in project or rel in external:
            continue
        p = root/rel if (root/rel).is_file() else repo/rel
        if not p.is_file():
            obj = Path(rel).with_suffix('.olean')
            choices = [Path('/tmp/veritile-lean-toolchain/lib/lean')/obj]
            choices += [d/'.lake/build/lib/lean'/obj for d in (repo/'.lake/packages').iterdir()]
            q = next((q for q in choices if q.is_file()),None)
            assert q,rel
            external[rel] = {'olean':str(q.resolve()),'sha256':sha(q)}
            continue
        row = {'source':str(p),'sha256':sha(p),'package':p == root/rel}
        if p != root/rel:
            q = root/Path(rel).with_suffix('.olean')
            assert q.is_file() and q.is_symlink(),q
            row.update(olean=str(q.resolve()),olean_sha256=sha(q))
        project[rel] = row
        for line in p.read_text().splitlines():
            if line.startswith('import '):
                for name in line[7:].split():
                    if name.startswith('--'): break
                    queue.append(name.replace('.','/')+'.lean')
    inputs = [repo/'lakefile.toml',repo/'lake-manifest.json',repo/'lean-toolchain',
      repo/'bench/original_program_acceptance.md',repo/(prefix+'vector_addition.py'),
      Path('/tmp/veritile-lean-toolchain/bin/lean'),Path('/tmp/veritile-lean-toolchain/bin/lake')]
    inputs += sorted((repo/prefix).rglob('*.lean'))
    inputs += [audit/n for n in ['rows.json','audit-result.json','probe_receipt.json','frontend_matrix.json','promotion_matrix.json']]
    inputs += [base/'acceptance-standard-calibration'/n for n in ['receipt.json','calibration.json','evidence.json']]
    for package in ['vector-addition-wip','vector-addition-dtype-extension-wip',
       'vector-addition-integer-words-wip','vector-addition-mixed-fp32-revision1-wip',
       'vector-addition-mixed-half-bfloat-wip','vector-addition-fp64-core-wip',
       'vector-addition-fp64-grid-wip','vector-dtype-admission-wip']:
        inputs.append(base/package/'evidence/receipt.json')
    inputs += [base/'vector-addition-fp64-core-wip/evidence/numerical-correspondence.json']
    return {'project':dict(sorted(project.items())),'external_objects':dict(sorted(external.items())),
            'input_sha256':{str(p):sha(p) for p in inputs}}

def source_check():
    source = repo/(prefix+'vector_addition.py')
    observed = read(audit/'audit-result.json')
    assert sha(source) == observed['files']['original_python']['sha256']
    tree = ast.parse(source.read_text())
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    statements = {name:[ast.unparse(n) for n in functions[name].body] for name in ['add_kernel','add']}
    assert statements['add_kernel'] == [
      'pid = tl.program_id(axis=0)','block_start = pid * BLOCK_SIZE',
      'offsets = block_start + tl.arange(0, BLOCK_SIZE)','mask = offsets < n_elements',
      'x = tl.load(x_ptr + offsets, mask=mask)','y = tl.load(y_ptr + offsets, mask=mask)',
      'output = x + y','tl.store(output_ptr + offsets, output, mask=mask)']
    assert statements['add'] == [
      'output = torch.empty_like(x)','assert x.is_cuda and y.is_cuda and output.is_cuda',
      'n_elements = output.numel()',"grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)",
      'add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)','return output']
    rows = read(audit/'rows.json')
    assert len(rows) == len({(r['x'],r['y']) for r in rows}) == 289
    stages = Counter(r['finite_lean_model']['stage'] for r in rows)
    assert stages == {'admitted':189,'signature':64,'addition':36}
    admitted = [r for r in rows if r['finite_lean_model']['stage']=='admitted']
    counts = Counter(family(r['x'],r['y']) for r in admitted)
    assert counts == {'integer':81,'floating':25,'mixed32':18,'mixedSmall':36,'wide':29}
    execution = (root/modules[1]).read_text()
    for name in ['IntegerSource.runGrid','TypedFloat.runGrid','MixedFP32.runGrid','MixedSmall.runGrid','FP64Memory.runGrid']:
        assert name in execution
    assert 'runGrid M p' not in execution
    result = {'original_source':str(source),'source_sha256':sha(source),'source_statements':statements,
      'inherited_admission_audit':str(audit/'audit-result.json'),'inherited_audit_sha256':sha(audit/'audit-result.json'),
      'stage_counts':dict(stages),'selected_execution_family_counts':dict(counts),
      'new_frontend_runs':0,'native_backend_runs':0,'gpu_runs':0,
      'source_binding':'Existing source-corresponding host skeleton, exact finite dtype dispatch, direct existing grid invocation and reviewed primitive outcomes.',
      'not_claimed':'No formal compiler or native implementation proof; no unrestricted undefined-conversion value domain.'}
    dump(root/'evidence/source-route-check.json',result)

ints = {'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
formats = {'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64',
           'float8_e4m3fn':'f8e4','float8_e5m2':'f8e5'}

def family(x,y):
    if 'float64' in (x,y): return 'wide'
    if x in ints and y in ints: return 'integer'
    if x not in ints and y not in ints: return 'floating'
    return 'mixed32' if 'float32' in (x,y) else 'mixedSmall'

def binding(port, operation, statement, module, configuration=None):
    p = repo/(prefix+'Proofs/'+module+'.lean')
    return {'port':port,'operation':operation,'statement':statement,'source':str(p),'source_sha256':sha(p),
            'configuration':configuration or {},'status':'proposed deferred numerical correspondence'}

def profile_operations(row):
    x,y = row['x'],row['y']; f=family(x,y)
    ops=[]
    def add(port,op,stmt,module,config=None): ops.append(binding(port,op,stmt,module,config))
    if f=='integer': return ops
    if f=='floating':
        comp=row['finite_lean_model']['promotion_result']
        comp={'fp8e4nv':'f8e4','fp8e5':'f8e5'}.get(comp,comp)
        for label,d in [('x',x),('y',y)]:
            if formats[d]!=comp:
                add('b.rest.cast none '+formats[d]+' '+comp,
                    formats[d]+' -> '+comp+' input promotion','output = x + y','TypedFloat',{'operand':label})
        add('b.rest.binary none '+comp+' .add',comp+' addition','output = x + y','TypedFloat')
        if comp!=formats[x]:
            add('b.rest.cast none '+comp+' '+formats[x],comp+' -> '+formats[x]+' output conversion',
                'tl.store(output_ptr + offsets, output, mask=mask)','TypedFloat',
                {'rounding':'RTNE for fp8 destination; exact source-native mode otherwise'})
        return ops
    if f in ('mixed32','mixedSmall'):
        integer=x if x in ints else y
        floating=y if x in ints else x
        first=x in ints
        width=8 if integer=='bool' else int(re.search(r'\d+',integer).group())
        signed=integer.startswith('int') or integer=='bool'
        compute='fp16' if floating=='float16' else 'fp32'
        module='MixedFP32Source' if f=='mixed32' else 'MixedSmallSource'
        add(('MixedFP32.integerRounded' if f=='mixed32' else 'MixedSmall.integerRounded'),
            'direct integer/bool -> '+compute+' rounding','output = x + y',module,
            {'input_width':1 if integer=='bool' else width,'integer_signed':integer.startswith('int'),'cfg':'b.cfg'})
        if floating=='bfloat16':
            add('MixedSmall.widenBFloat','bf16 -> fp32 field extension','output = x + y',module)
        add('FP.add' if compute=='fp32' else 'MixedSmall.addTarget .fp16',
            compute+' addition','output = x + y',module,{'cfg':'b.cfg'})
        if first:
            if compute=='fp16':
                add('MixedSmall.widenHalf','fp16 -> fp32 before integer output',
                    'tl.store(output_ptr + offsets, output, mask=mask)',module)
            add('FP.IntegerConversion.'+('toSigned' if signed else 'toUnsigned'),
                'fp32 -> '+('signed' if signed else 'unsigned')+str(width)+' toward zero',
                'tl.store(output_ptr + offsets, output, mask=mask)',module,
                {'width':width,'domain':'ValueDomain.DefinedConversions; no undefined native result is assumed'})
        elif floating=='bfloat16':
            add('MixedSmall.narrowBFloat','fp32 -> bf16 output rounding',
                'tl.store(output_ptr + offsets, output, mask=mask)',module,{'cfg':'b.cfg'})
        return ops
    def promote(d,operand):
        if d=='float64' or d=='bool': return
        if d in ints:
            port='signedTo64' if d.startswith('int') else 'unsignedTo64'
            add('b.wide.'+port,d+' -> fp64','output = x + y','FP64TypedCore',{'operand':operand,'width':int(re.search(r'\d+',d).group())})
        elif d in ('float16','bfloat16'):
            port='halfTo32' if d=='float16' else 'bfloatTo32'
            add('b.wide.'+port,d+' -> fp32 extension','output = x + y','FP64TypedCore',{'operand':operand})
            add('b.wide.singleTo64','fp32 -> fp64 extension','output = x + y','FP64TypedCore',{'operand':operand})
        else:
            port={'float32':'singleTo64','float8_e4m3fn':'e4To64','float8_e5m2':'e5To64'}[d]
            add('b.wide.'+port,d+' -> fp64','output = x + y','FP64TypedCore',{'operand':operand})
    promote(x,'x');promote(y,'y')
    add('b.wide.add64','fp64 addition','output = x + y','FP64TypedCore')
    if x!='float64':
        config={}
        if x in ints:
            port='toSigned' if x=='bool' or x.startswith('int') else 'toUnsigned'
            config={'width':8 if x=='bool' else int(re.search(r'\d+',x).group()),
                    'domain':'Ready.OutputDefined, a genuine defined-value restriction on nine dtype routes'}
        else:
            port={'float16':'toHalf','bfloat16':'toBFloat','float32':'toSingle',
                  'float8_e4m3fn':'toE4RTNE','float8_e5m2':'toE5RTNE'}[x]
        add('b.wide.'+port,'direct fp64 -> '+x+' output conversion',
            'tl.store(output_ptr + offsets, output, mask=mask)','FP64TypedCore',config)
    return ops

def numerical_proposal():
    rows=read(audit/'rows.json')
    proposal=[]
    for row in rows:
        if row['finite_lean_model']['stage']!='admitted': continue
        x,y=row['x'],row['y']
        proposal.append({'x_dtype':x,'y_dtype':y,'output_dtype':x,'family':family(x,y),
          'compute_dtype':row['finite_lean_model']['promotion_result'],
          'native_store_dtype':row['finite_lean_model']['store'],
          'public_contracts':contracts[:2], 'operations':profile_operations(row),
          'configuration':{'block':1024,'length':'0 <= x.numel < 2^31; y.numel >= x.numel',
            'source':'original vector_addition.py','target':'frozen CUDA sm90/Triton 3.7.1 primitive domain',
            'value_domain':'TypedHost.ValueDomain; fp64 OutputDefined retained, mixed undefined policies proved irrelevant on defined active outputs'},
          'execution_binding':{'module':modules[1],'sha256':sha(root/modules[1]),'declaration':'TypedHost.execute'},
          'host_binding':{'module':modules[2],'sha256':sha(root/modules[2]),'declaration':'TypedHost.launch'}})
    assert len(proposal)==189
    fp64ports={op['port'] for r in proposal if r['family']=='wide' for op in r['operations']}
    assert len(fp64ports)==15,fp64ports
    dump(root/'evidence/numerical-bindings-proposal.json',{
      'integrated_in_repository_queue':False,'independent_registration_review':'pending root review',
      'native_numerical_validation':False,'new_arithmetic_primitives':0,
      'source_sha256':sha(repo/(prefix+'vector_addition.py')),
      'profile_count':189,'operation_binding_count':sum(len(r['operations']) for r in proposal),
      'fp64_callback_families':sorted(fp64ports),'profiles':proposal,
      'undefined_value_policy':'Never deferred as a native numerical equality. Public value domains exclude undefined active conversions; lower shared totalized semantics retain explicit arbitrary policies; wide faults are logical diagnostics only.',
      'integer_only':'Concrete finite-width arithmetic and bool byte policy; no new floating numerical port on 81 routes.',
      'trusted_scalar_bits':'Bool byte !=0, bool i1 zero/one conversion and signed8 byte-preserving bool stores remain concrete inherited operations.'})

def finalize():
    receipt=root/'evidence/receipt.json'; assert not receipt.exists()
    assert read(root/'evidence/dependencies-before.json')==read(root/'evidence/dependencies-after.json')==dependencies()
    history=read(root/'evidence/verification-attempts.json'); results=read(root/'evidence/verification-results.json')
    assert [r['module'] for r in results]==modules and history[-5:]==results
    for row in history:
        assert row['source_sha256_before']==row['snapshot_sha256']==sha(root/row['source_snapshot'])
        assert row['log_sha256']==sha(root/row['log'])
    for row in results:
        assert row['exit_code']==0
        assert row['source_sha256_before']==row['source_sha256_after']==sha(root/row['module'])
        assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',(root/row['log']).read_text())
    source_check(); numerical_proposal()
    scope={'status':'compiled additive typed host composition; original-program acceptance pending',
      'acceptance_increment':0,'new_dtype_route_increment':0,'composed_admitted_dtype_pairs':189,
      'signature_rejections':64,'addition_rejections':36,'existing_route_sources_rewritten':False,
      'public_contracts':contracts,'metadata_contract':'dtype/shape/numel/device/CUDA/offset, output fresh from both inputs, both carrier input frames',
      'value_domain':'Explicit finite-word/tag views, exact defined active mixed-output conversions, fp64 OutputDefined for nine integer/bool routes; 20 floating-output fp64 routes automatically defined.',
      'remaining_finite_obligations':['Pointer/view/stride/storage-offset interpretation and finite address/no-wrap domain binding',
       'Unsafe source inputs outside SourceDomain and undefined active conversions outside ValueDomain need separate source outcome coverage',
       'Reviewed source-grounded frontend/API/target domain binding beyond inherited supporting evidence',
       'Exact numerical queue integration, source-stage/schema2 registration and independent substantive root review'],
      'trusted_boundary':['Python/Torch empty_like metadata and freshness contract','Triton typed load/store and finite source graph interpretation',
        'Frozen-target first frontend dtype rejection classification','Complete ordered logical grid execution'],
      'native_execution':False,'repository_modified':False,'git_commands_run':False,
      'harness':'Original generic helper/module semantics remain available; no generic RNG/runtime/harness implementation proof requirement added.'}
    dump(root/'evidence/scope-review.json',scope)
    artifacts=['README.md','verify.py','freeze.py','evidence/source-route-check.json','evidence/numerical-bindings-proposal.json',
      'evidence/scope-review.json','evidence/dependencies-before.json','evidence/dependencies-after.json',
      'evidence/verification-attempts.json','evidence/verification-results.json']
    dump(receipt,{'schema_version':1,'frozen':True,'acceptance_increment':0,'new_dtype_route_increment':0,
      'modules':{p:sha(root/p) for p in modules},
      'module_objects':{str(Path(p).with_suffix('.olean')):sha((root/p).with_suffix('.olean')) for p in modules},
      'artifacts':{p:sha(root/p) for p in artifacts},'verification_attempt_count':len(history),
      'failed_attempts_preserved':sum(r['exit_code']!=0 for r in history),'final_clean_attempts':results,
      'independent_review_status':'pending root review of frozen package'})
    print(json.dumps({'receipt':str(receipt),'receipt_sha256':sha(receipt),'final_modules':5,'attempts':len(history)}))

mode=sys.argv[1]
if mode in ('before','after'):
    current=dependencies()
    if mode=='after': assert read(root/'evidence/dependencies-before.json')==current
    dump(root/('evidence/dependencies-'+mode+'.json'),current)
    print(json.dumps({'mode':mode,'project_sources':len(current['project']),'external_objects':len(current['external_objects'])}))
elif mode=='finalize': finalize()
else: raise SystemExit('Expected before, after or finalize')
