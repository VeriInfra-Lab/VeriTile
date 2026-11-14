"""Fixed-budget matrix-dot contraction with independent whole-tile observations.

The reference uses an FP32 intermediate between two launches: Triton's dot/add
combiner otherwise contracts both source expressions even with fusion disabled.
Raw bundles retain per-part statistics, fixed audits, sources and actual PTX.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time

import numpy as np

if __package__:
    from . import check_numerics as common, two_gates_statistics as stats
else:
    import check_numerics as common
    import two_gates_statistics as stats

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'experiments/floating_point/dot'
FORMAT = 'veritile-two-gates-dot'


def digest(value):
    return hashlib.sha256(common.registry.canonical_json(value)).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources():
    names = ['scripts/two_gates_dot.py', 'scripts/two_gates_statistics.py', 'scripts/check_two_gates.py',
             'scripts/check_numerics.py', 'scripts/numerical_registry.py',
             'experiments/floating_point/two_gates_config.py',
             'experiments/floating_point/dot/config.py', 'experiments/floating_point/dot/kernels.py',
             'experiments/floating_point/dot/rules.json', 'experiments/floating_point/requirements.txt',
             'experiments/floating_point/requirements-cpu.txt']
    return {name: file_hash(ROOT/name) for name in names}


def plan(smoke=False, samples=None):
    settings = dict(common.load_module(ROOT/'experiments/floating_point/two_gates_config.py').SETTINGS)
    profile = common.load_module(DIRECTORY/'config.py').PROFILE
    if profile['rule'] not in common.registry.load_catalog(DIRECTORY/'rules.json'):
        raise ValueError('unregistered local matrix-dot rule')
    settings['workers'] = 1
    if samples is not None:
        if type(samples) is not int or not 2 <= samples <= 2**53:
            raise ValueError('sample count must be an integer in [2,2^53]')
        settings['samples'] = samples
    if smoke:
        settings['samples'] = 33
        profile = {**profile, 'groups_per_part': 16}
    return {'settings': settings, 'profile': profile, 'smoke': smoke,
            'diagnostic': smoke or settings['samples'] != 2**32,
            'seeds': {name: int(digest([settings['seed'], 'DOT-ADD-TO-DOTACC', name])[:15],16)
                      for name in ['a','b','acc']},
            'formal_domain': 'all FP32 payloads; padded zero lanes; arbitrary live accumulator',
            'probe_domain': 'independent quantized Normal(1,1) entries; one whole tile is one group',
            'reference_execution': 'standalone FP32 dot stored to FP32, then separate FP32 add',
            'oracle_order': 'FP64 products; K=0..31 sequential additions from +0; add live accumulator last'}


def part_counts(p):
    n, width = p['settings']['samples'], p['profile']['groups_per_part']
    return [min(width,n-start) for start in range(0,n,width)]


def audit_indices(p):
    count = len(part_counts(p))
    return set(np.linspace(0,count-1,min(count,p['settings']['audit_parts']),dtype=np.int64).tolist())


def summarize(observations):
    x = np.asarray(observations)
    if (x.ndim != 2 or x.shape[1] != 4 or x.dtype != np.float64 or
            not np.isfinite(x).all() or not np.isin(x[:,3],[0.,1.]).all() or
            (x[:,1:3] < 0).any() or (x[x[:,3]==0,:3] != 0).any()):
        raise ValueError('invalid independent whole-tile observations')
    valid = x[:,3] == 1; values = x[valid,0]
    n = int(valid.sum()); mean = float(values.mean()) if n else 0.
    m2 = float(np.square(values-mean).sum())
    ep,eq = (x[:,j].max(initial=0.) for j in [1,2])
    return np.array([n,mean,m2,ep,eq,len(x),n,np.abs(values).max(initial=0.),n,ep,eq,len(x)],dtype=np.float64)


def cpu_observations(a,b,c,p,q,o,u):
    """Check FP64 oracle, ULPs, all output lanes and whole-group correlation."""
    if any(x.shape != a.shape for x in [b,c,p,q,o,u]) or a.ndim != 3 or a.shape[1:] != (32,32):
        raise ValueError('raw tile audit shapes differ')
    exact = np.zeros(a.shape,dtype=np.float64)
    for k in range(32):
        exact += a[:,:,k,None].astype(np.float64)*b[:,None,k,:].astype(np.float64)
    exact += c.astype(np.float64)
    if not np.array_equal(exact,o,equal_nan=True):
        raise ValueError('GPU oracle differs from independent ordered FP64 evaluation')
    with np.errstate(over='ignore',invalid='ignore'):
        rounded = np.minimum(np.abs(o),np.finfo(np.float32).max).astype(np.float32)
        upper = np.nextafter(rounded,np.float32(np.inf))
        lower = np.nextafter(rounded,np.float32(0.))
        expected_u = np.where(np.isfinite(rounded)&np.isinf(upper),rounded.astype(np.float64)-lower,
                              upper.astype(np.float64)-rounded)
    if not np.array_equal(expected_u,u,equal_nan=True):
        raise ValueError('GPU ULP differs from shared output-format spacing')
    finite = np.isfinite(p)&np.isfinite(q)&np.isfinite(o)
    valid = finite.all(axis=(1,2))
    with np.errstate(invalid='ignore'):
        delta = (q.astype(np.float64)-p.astype(np.float64))/u
        return np.stack([np.where(valid,delta.mean(axis=(1,2)),0.),
            np.where(valid,np.abs(p.astype(np.float64)-o).max(axis=(1,2)),0.),
            np.where(valid,np.abs(q.astype(np.float64)-o).max(axis=(1,2)),0.),valid.astype(np.float64)],axis=1)


def classifications(x):
    x = np.asarray(x,dtype=np.float32)
    return np.select([np.isnan(x),np.isposinf(x),np.isneginf(x),x==0],
                     [0,1,2,3+np.signbit(x).astype(np.int8)],default=5).astype(np.int8)


def check_lowerings(ptx):
    if (not all('.tf32.tf32.' in ptx[name] and 'mma.sync' in ptx[name]
                for name in ['reference-dot','candidate']) or
            not re.search(r'\badd\.(?:rn\.)?f32\b',ptx['reference-add']) or
            'mma.sync' in ptx['reference-add']):
        raise ValueError('required separate FP32 dot/add or seeded TF32 dotAcc lowering is absent')


def launch(torch,k,inputs,save=False):
    n = inputs[0].shape[0]; launch = dict(num_warps=4,num_stages=3,enable_fp_fusion=False)
    p,q,t = [torch.empty_like(inputs[0]) for _ in range(3)]
    o,u = [torch.empty(inputs[0].shape if save else (1,),device='cuda',dtype=torch.float64) for _ in range(2)]
    obs = torch.empty((n,4),device='cuda',dtype=torch.float64)
    handles = [k.reference_dot[(n,)](*inputs[:2],t,**launch),
               k.reference_add[(n,)](t,inputs[2],p,**launch),
               k.candidate[(n,)](*inputs,q,**launch),
               k.observer[(n,)](*inputs,p,q,o,u,obs,save,**launch)]
    return (p,q,o,u,obs),{name:h.asm['ptx'] for name,h in zip(
        ['reference-dot','reference-add','candidate','observer-audit' if save else 'observer'],handles)}


def special_probe(torch,k,output):
    values = [0.,-0.,float('inf'),-float('inf'),float('nan'),np.finfo(np.float32).max,
              -np.finfo(np.float32).max,float(np.nextafter(np.float32(0),np.float32(1)))]
    arrays=[];labels=[]
    for role in range(3):
        for value in values:
            group = np.ones((3,32,32),dtype=np.float32)
            group[role].fill(value)
            if role == 2:
                group[0].fill(0.)
            arrays.append(group);labels.append({'operand':['a','b','acc'][role],'value':str(value)})
    # Preserve zero padding and both signed-zero inputs explicitly.
    for sign in [0.,-0.]:
        group=np.zeros((3,32,32),dtype=np.float32);group[0].fill(sign);group[2].fill(-0.)
        arrays.append(group);labels.append({'operand':'zero-padding','value':str(sign)})
    x=np.stack(arrays); inputs=[torch.from_numpy(x[:,j].copy()).cuda() for j in range(3)]
    (p,q,*_),_=launch(torch,k,inputs)
    p,q=p.cpu().numpy(),q.cpu().numpy()
    np.savez_compressed(output/'classifications.npz',inputs=x,reference=p,candidate=q)
    mismatch=classifications(p)!=classifications(q)
    payload=(np.isnan(p)&np.isnan(q))&(p.view(np.uint32)!=q.view(np.uint32))
    return {'groups':len(x),'classification_mismatches':int(mismatch.sum()),
            'nan_payload_differences':int(payload.sum()),'cases':[
                {**label,'classification_mismatches':int(mismatch[i].sum()),'nan_payload_differences':int(payload[i].sum())}
                for i,label in enumerate(labels)],
            'classification_scope':'NaN, signed infinity, signed zero and finite nonzero; payload bits recorded separately',
            'all_words_scope':'BLOCKED' if mismatch.any() or payload.any() else 'NO_DIAGNOSTIC_MISMATCH; external model premise retained'}


def replay(bundle):
    m=json.loads((bundle/'manifest.json').read_text());record=json.loads((bundle/'record.json').read_text())
    if m['format']!=FORMAT or record.get('state')!='COMPLETE' or record['manifest_sha256']!=digest(m):
        raise ValueError('incomplete or mismatched matrix-dot bundle')
    expected=plan(m['smoke'],m['settings']['samples'])
    if any(m.get(key)!=value for key,value in expected.items()) or set(m['sources'])!=set(sources()):
        raise ValueError('matrix-dot precision, source set or fixed protocol differs')
    for name,h in m['sources'].items():
        if file_hash(bundle/'sources'/name)!=h:
            raise ValueError('source snapshot hash differs: '+name)
    for name,h in record['files'].items():
        if file_hash(bundle/name)!=h: raise ValueError('evidence file hash differs: '+name)
    data=np.load(bundle/'statistics.npz',allow_pickle=False);rows=data['rows'];drawn=data['drawn']
    counts=part_counts(m)
    required={'statistics.npz','fingerprints.json','classifications.npz','reference-dot.ptx',
              'reference-add.ptx','candidate.ptx','observer-audit.ptx',
              *(f'audit-{j:06d}.npz' for j in audit_indices(m))}
    if len(audit_indices(m))<len(counts): required.add('observer.ptx')
    if set(record['files'])!=required:
        raise ValueError('missing or unexpected required dot evidence files')
    check_lowerings({name:(bundle/(name+'.ptx')).read_text() for name in
                     ['reference-dot','reference-add','candidate']})
    if rows.shape!=(len(counts),len(stats.FIELDS)) or drawn.tolist()!=counts:
        raise ValueError('fixed whole-group budget or part coverage differs')
    if len(json.loads((bundle/'fingerprints.json').read_text()))!=len(counts):
        raise ValueError('missing input/output fingerprints')
    if {audit['part'] for audit in record['audits']}!=audit_indices(m) or len(record['audits'])!=len(audit_indices(m)):
        raise ValueError('mandatory raw tile audit coverage differs')
    for audit in record['audits']:
        a=np.load(bundle/audit['file'],allow_pickle=False);j=audit['part']
        if a['observations'].shape!=(counts[j],4) or not np.allclose(summarize(a['observations']),rows[j],rtol=2e-13,atol=1e-13):
            raise ValueError('raw group observations differ from part statistics')
        checked=cpu_observations(*(a[name] for name in ['a','b','acc','p','q','oracle','ulp']))
        if not np.allclose(checked,a['observations'][a['indices']],rtol=2e-13,atol=1e-13):
            raise ValueError('raw output lanes differ from GPU whole-group observer')
    special=np.load(bundle/'classifications.npz',allow_pickle=False)
    cm=int(np.count_nonzero(classifications(special['reference'])!=classifications(special['candidate'])))
    pp,qq=special['reference'],special['candidate']
    pm=int(np.count_nonzero((np.isnan(pp)&np.isnan(qq))&(pp.view(np.uint32)!=qq.view(np.uint32))))
    if cm!=record['special']['classification_mismatches'] or pm!=record['special']['nan_payload_differences']:
        raise ValueError('special-value classification report differs')
    result=stats.evaluate_parts(rows,np.zeros(len(rows),dtype=np.int8),drawn,[1.],m['settings'],m['diagnostic'])
    if result!=record['result']:
        raise ValueError('CPU two-gates decisions differ from recorded decisions')
    if not m['diagnostic'] and int(drawn.sum())!=2**32:
        raise ValueError('formal matrix-dot run did not reach 2^32 whole groups')
    return {'complete':True,'rule':m['profile']['rule'],'manifest_sha256':digest(m),
            'result':result,'special':record['special'],'backend':m['backend'],
            'all_words_admission_eligible':result['decision']=='ACCEPT' and cm==0 and pm==0,
            'raw_audits_verified':len(record['audits']),'part_statistics_verified':len(rows),
            'source_hashes':m['sources'],'lowerings':{name:h for name,h in record['files'].items() if name.endswith('.ptx')},
            'limitations':'Numerical acceptance is a scoped modeling assumption; not an IEEE equality theorem or GPU conformance proof.'}


def run(args):
    import torch,triton
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError('matrix-dot experiment requires an NVIDIA CUDA GPU')
    p=plan(args.smoke,args.samples);out=args.output;out.mkdir(parents=True,exist_ok=False)
    backend={'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
             'torch':str(torch.__version__),'triton':triton.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    backend['drivers']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    m={'format':FORMAT,**p,'backend':backend,'sources':sources()}
    for name in m['sources']:
        path=out/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    common.write_json(out/'manifest.json',m)
    record={'state':'RUNNING','manifest_sha256':digest(m)};common.write_json(out/'record.json',record)
    start=time.monotonic();last=start
    try:
        k=common.load_module(DIRECTORY/'kernels.py')
        record['special']=special_probe(torch,k,out)
        gens={name:torch.Generator(device='cuda').manual_seed(seed) for name,seed in p['seeds'].items()}
        counts=part_counts(p);audits=audit_indices(p);saved=[];fingerprints=[];lowerings={}
        rows=np.empty((len(counts),len(stats.FIELDS)),dtype=np.float64);completed=0
        torch.cuda.reset_peak_memory_stats()
        for j,n in enumerate(counts):
            inputs=[torch.randn((n,32,32),device='cuda',dtype=torch.float64,generator=gens[name]).add_(1).float() for name in ['a','b','acc']]
            (pv,qv,o,u,obs),ptx=launch(torch,k,inputs,j in audits)
            for name,text in ptx.items():
                if name in lowerings and lowerings[name]!=hashlib.sha256(text.encode()).hexdigest():
                    raise ValueError('compiled PTX changed during fixed run: '+name)
                if name not in lowerings:
                    (out/(name+'.ptx')).write_text(text);lowerings[name]=file_hash(out/(name+'.ptx'))
            check_lowerings(ptx)
            observations=obs.cpu().numpy();rows[j]=summarize(observations)
            completed+=n
            fingerprints.append(hashlib.sha256(observations.tobytes()).hexdigest())
            if j in audits:
                indices=np.unique(np.linspace(0,n-1,min(n,32),dtype=np.int64))
                raw={name:value[indices].cpu().numpy() for name,value in zip(['a','b','acc','p','q','oracle','ulp'],[*inputs,pv,qv,o,u])}
                checked=cpu_observations(*(raw[name] for name in ['a','b','acc','p','q','oracle','ulp']))
                if not np.allclose(checked,observations[indices],rtol=2e-13,atol=1e-13):
                    raise ValueError('GPU group observer failed independent CPU check')
                name=f'audit-{j:06d}.npz';np.savez_compressed(out/name,observations=observations,indices=indices,**raw)
                saved.append({'part':j,'file':name})
            now=time.monotonic()
            if now-last>=5 or j+1==len(counts):
                progress={'groups_completed':completed,'groups_total':p['settings']['samples'],
                          'parts_completed':j+1,'parts_total':len(counts),'elapsed_seconds':now-start}
                common.write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True);last=now
        result=stats.evaluate_parts(rows,np.zeros(len(rows),dtype=np.int8),np.array(counts,dtype=np.int64),[1.],p['settings'],p['diagnostic'])
        np.savez_compressed(out/'statistics.npz',rows=rows,drawn=np.array(counts,dtype=np.int64))
        common.write_json(out/'fingerprints.json',fingerprints)
        names=['statistics.npz','fingerprints.json','classifications.npz',*(a['file'] for a in saved),*(name+'.ptx' for name in lowerings)]
        record.update(state='COMPLETE',result=result,audits=saved,files={name:file_hash(out/name) for name in names},
                      elapsed_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        common.write_json(out/'record.json',record)
        report=replay(out);common.write_json(out/'report.json',report)
        print(json.dumps(report,indent=2));return 0
    except BaseException as error:
        record.update(state='ERROR',reason=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-start)
        common.write_json(out/'record.json',record);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['check','run']:
        p=sub.add_parser(command);p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int)
        if command=='run': p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('import');p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='check': print(json.dumps(plan(args.smoke,args.samples),indent=2));return 0
    if args.command=='run': return run(args)
    report=replay(args.bundle)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(report,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())
