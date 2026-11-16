"""Fixed-budget FP64 dot/add gates with independently replayable whole-tile data.

The FP32 experiment and its rejected evidence remain separate and unchanged.
FP64 statistical acceptance cannot establish an FP32 rule or a kernel refutation.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

if __package__:
    from . import two_gates_dot as dot, two_gates_statistics as stats
else:
    import two_gates_dot as dot
    import two_gates_statistics as stats

ROOT = dot.ROOT
DIRECTORY = ROOT/'experiments/floating_point/dot/fp64'
FORMAT = 'veritile-two-gates-dot-fp64'


def sources():
    names = [*dot.sources(), *(str(path.relative_to(ROOT)) for path in dot.common.SOURCES),
             'scripts/two_gates_dot_fp64.py',
             'experiments/floating_point/dot/fp64/config.py',
             'experiments/floating_point/dot/fp64/kernels.py']
    return {name: dot.file_hash(ROOT/name) for name in names}


def plan(smoke=False, samples=None):
    p = dot.plan(smoke, samples)
    p['profile'] = dict(dot.common.load_module(DIRECTORY/'config.py').PROFILE)
    if smoke:
        p['profile']['groups_per_part'] = 16
    p.update(formal_domain='all FP64 payloads; padded zero lanes; arbitrary live accumulator',
             probe_domain='independent FP64 Normal(1,1) entries; one whole tile is one group',
             reference_execution='standalone FP64 dot stored to FP64, then separate FP64 add',
             oracle_order='double-double products and K sum; add live accumulator last; exact rational CPU audit')
    return p


def classifications(x):
    x = np.asarray(x, dtype=np.float64)
    return np.select([np.isnan(x), np.isposinf(x), np.isneginf(x), x == 0],
                     [0, 1, 2, 3+np.signbit(x).astype(np.int8)], default=5).astype(np.int8)


def exact_audit(a, b, c, hi, lo, ulp):
    """Exact integer dot products; no reuse of the GPU oracle arithmetic."""
    max_error = 0.
    for g in range(len(a)):
        ratios = [[v.as_integer_ratio() for v in values[g].flat] for values in (a, b, c)]
        exponent = max(d.bit_length()-1 for group in ratios for _, d in group)
        encoded = [np.array([n << (exponent-(d.bit_length()-1)) for n, d in group],
                            dtype=object).reshape(32, 32) for group in ratios]
        exact = encoded[0] @ encoded[1] + encoded[2]*(1 << exponent)
        denominator = 1 << (2*exponent)
        for i in range(32):
            for j in range(32):
                value = Fraction(int(exact[i,j]), denominator)
                stored = Fraction(float(hi[g,i,j]))+Fraction(float(lo[g,i,j]))
                rounded = np.float64(abs(value))
                expected_ulp = np.nextafter(rounded, np.float64(np.inf))-rounded
                if expected_ulp != ulp[g,i,j]:
                    raise ValueError('FP64 ULP differs from exact rational source reference')
                error = float(abs(value-stored)/Fraction(float(expected_ulp)))
                max_error = max(max_error, error)
                if error >= 2.**-35:
                    raise ValueError('double-double oracle differs from exact rational source reference')
    return max_error


def cpu_observations(a, b, c, p, q, hi, lo, ulp, exact=True):
    values = (a,b,c,p,q,hi,lo,ulp)
    if (a.ndim != 3 or a.shape[1:] != (32,32) or any(x.shape != a.shape or x.dtype != np.float64 for x in values)):
        raise ValueError('raw FP64 tile audit dtype/shape differs')
    if not all(np.isfinite(x).all() for x in (a,b,c,hi,lo,ulp)) or (ulp <= 0).any():
        raise ValueError('invalid finite reference audit')
    error = exact_audit(a,b,c,hi,lo,ulp) if exact else None
    valid = (np.isfinite(p)&np.isfinite(q)).all(axis=(1,2))
    delta = (q-p)/ulp
    ep,eq = np.abs((p-hi)-lo),np.abs((q-hi)-lo)
    obs = np.stack([np.where(valid,delta.mean(axis=(1,2)),0.),
                    np.where(valid,ep.max(axis=(1,2)),0.),
                    np.where(valid,eq.max(axis=(1,2)),0.),valid.astype(np.float64)],axis=1)
    return obs,error


def launch(torch, k, inputs, save=False, observe=True):
    n = len(inputs[0]); options = dict(plan()['profile']['launch'])
    p,q,t = [torch.empty_like(inputs[0]) for _ in range(3)]
    handles = [k.separate_dot[(n,)](*inputs[:2],t,**options),
               k.separate_add[(n,)](t,inputs[2],p,**options),
               k.seeded_dot[(n,)](*inputs,q,**options)]
    if not observe:
        return p,q
    hi,lo,u = [torch.empty_like(inputs[0]) if save else torch.empty(1,device='cuda',dtype=torch.float64) for _ in range(3)]
    obs = torch.empty((n,4),device='cuda',dtype=torch.float64)
    handles.append(k.observer[(n,)](*inputs,p,q,hi,lo,u,obs,save,**options))
    if any(v.dtype != torch.float64 for v in (*inputs,p,q)):
        raise ValueError('FP64 source pair was narrowed')
    compiled = {name:hashlib.sha256(handle.asm['cubin']).hexdigest() for name,handle in zip(
        ['separate-dot','separate-add','seeded-dot','observer-audit' if save else 'observer'],handles)}
    return (p,q,hi,lo,u,obs),compiled


def special_probe(torch,k,out):
    values = [0.,-0.,float('inf'),-float('inf'),float('nan'),np.finfo(np.float64).max,
              -np.finfo(np.float64).max,float(np.nextafter(np.float64(0),np.float64(1)))]
    arrays=[];labels=[]
    for role in range(3):
        for value in values:
            group=np.ones((3,32,32),np.float64);group[role].fill(value)
            if role==2:group[0].fill(0.)
            arrays.append(group);labels.append({'operand':['a','b','acc'][role],'value':str(value)})
    for sign in [0.,-0.]:
        group=np.zeros((3,32,32),np.float64);group[0].fill(sign);group[2].fill(-0.)
        arrays.append(group);labels.append({'operand':'zero-padding','value':str(sign)})
    x=np.stack(arrays);pv,qv=launch(torch,k,[torch.from_numpy(x[:,j].copy()).cuda() for j in range(3)],observe=False)
    p,q=pv.cpu().numpy(),qv.cpu().numpy()
    np.savez_compressed(out/'classifications.npz',inputs=x,reference=p,candidate=q)
    cm=classifications(p)!=classifications(q)
    pm=(np.isnan(p)&np.isnan(q))&(p.view(np.uint64)!=q.view(np.uint64))
    return {'groups':len(x),'classification_mismatches':int(cm.sum()),'nan_payload_differences':int(pm.sum()),
            'cases':[{**label,'classification_mismatches':int(cm[i].sum()),'nan_payload_differences':int(pm[i].sum())}
                     for i,label in enumerate(labels)],
            'scope':'Special-value observations are diagnostic, not an exhaustive all-word proof.'}


def replay(bundle):
    m=json.loads((bundle/'manifest.json').read_text());record=json.loads((bundle/'record.json').read_text())
    expected=plan(m['smoke'],m['settings']['samples'])
    if (m['format']!=FORMAT or record.get('state')!='COMPLETE' or record['manifest_sha256']!=dot.digest(m)
            or any(m.get(key)!=value for key,value in expected.items()) or set(m['sources'])!=set(sources())):
        raise ValueError('FP64 bundle identity, fixed precision/budget or scope differs')
    for name,h in m['sources'].items():
        if dot.file_hash(bundle/'sources'/name)!=h:raise ValueError('frozen source hash differs: '+name)
    for name,h in record['files'].items():
        if dot.file_hash(bundle/name)!=h:raise ValueError('evidence hash differs: '+name)
    counts=dot.part_counts(m);audits=dot.audit_indices(m)
    required={'statistics.npz','fingerprints.json','classifications.npz','compiled.json',
              *(f'audit-{j:06d}.npz' for j in audits)}
    if set(record['files'])!=required:raise ValueError('missing or unexpected mandatory FP64 evidence')
    data=np.load(bundle/'statistics.npz',allow_pickle=False);rows=data['rows'];drawn=data['drawn']
    if rows.shape!=(len(counts),len(stats.FIELDS)) or drawn.tolist()!=counts:raise ValueError('fixed whole-group coverage differs')
    if len(json.loads((bundle/'fingerprints.json').read_text()))!=len(counts):raise ValueError('missing group fingerprints')
    if {a['part'] for a in record['audits']}!=audits or len(record['audits'])!=len(audits):raise ValueError('raw audit coverage differs')
    max_error=0.
    for audit in record['audits']:
        a=np.load(bundle/audit['file'],allow_pickle=False);j=audit['part']
        if a['observations'].shape!=(counts[j],4) or not np.allclose(dot.summarize(a['observations']),rows[j],rtol=2e-13,atol=1e-13):
            raise ValueError('raw group moments differ')
        observed,error=cpu_observations(*(a[name] for name in ['a','b','acc','p','q','hi','lo','ulp']))
        if not np.allclose(observed,a['observations'][a['indices']],rtol=2e-13,atol=1e-13):raise ValueError('raw lanes differ from group observer')
        max_error=max(max_error,error)
    data=np.load(bundle/'classifications.npz',allow_pickle=False);pp,qq=data['reference'],data['candidate']
    if pp.dtype!=np.float64 or qq.dtype!=np.float64:raise ValueError('special-value precision differs')
    cm=int(np.count_nonzero(classifications(pp)!=classifications(qq)))
    pm=int(np.count_nonzero((np.isnan(pp)&np.isnan(qq))&(pp.view(np.uint64)!=qq.view(np.uint64))))
    if cm!=record['special']['classification_mismatches'] or pm!=record['special']['nan_payload_differences']:
        raise ValueError('special-value report differs')
    result=stats.evaluate_parts(rows,np.zeros(len(rows),np.int8),drawn,[1.],m['settings'],m['diagnostic'])
    if result!=record['result']:raise ValueError('CPU decisions differ from recorded gates')
    if not m['diagnostic'] and int(drawn.sum())!=2**32:raise ValueError('formal run did not reach 2^32 whole groups')
    return {'complete':True,'precision':'fp64','rule':m['profile']['rule'],'manifest_sha256':dot.digest(m),
            'result':result,'special':record['special'],'backend':m['backend'],
            'all_words_admission_eligible':result['decision']=='ACCEPT' and cm==0 and pm==0,
            'raw_audits_verified':len(audits),'part_statistics_verified':len(rows),
            'exact_rational_output_audits':sum(len(np.load(bundle/a['file'])['indices'])*1024 for a in record['audits']),
            'max_audited_reference_error_ulp':max_error,'source_hashes':m['sources'],
            'limitations':'FP64 numerical acceptance supplies no FP32 rule and does not disprove a whole kernel.'}


def run(args):
    import torch,triton
    if not torch.cuda.is_available() or torch.version.hip is not None:raise ValueError('NVIDIA CUDA GPU required')
    p=plan(args.smoke,args.samples);out=args.output;out.mkdir(parents=True,exist_ok=False)
    backend={'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
             'torch':str(torch.__version__),'triton':triton.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    backend['drivers']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    m={'format':FORMAT,**p,'backend':backend,'sources':sources()}
    for name in m['sources']:
        path=out/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    dot.common.write_json(out/'manifest.json',m)
    record={'state':'RUNNING','manifest_sha256':dot.digest(m)};dot.common.write_json(out/'record.json',record)
    start=time.monotonic();last=start
    try:
        k=dot.common.load_module(DIRECTORY/'kernels.py');record['special']=special_probe(torch,k,out)
        gens={name:torch.Generator(device='cuda').manual_seed(seed) for name,seed in p['seeds'].items()}
        counts=dot.part_counts(p);audits=dot.audit_indices(p);saved=[];fingerprints=[];compiled={}
        rows=np.empty((len(counts),len(stats.FIELDS)),np.float64);completed=0
        torch.cuda.reset_peak_memory_stats()
        for j,n in enumerate(counts):
            inputs=[torch.randn((n,32,32),device='cuda',dtype=torch.float64,generator=gens[name]).add_(1) for name in ['a','b','acc']]
            (pv,qv,hi,lo,u,obs),binaries=launch(torch,k,inputs,j in audits)
            for name,h in binaries.items():
                if name in compiled and compiled[name]!=h:raise ValueError('executed implementation changed during fixed run: '+name)
                compiled[name]=h
            observations=obs.cpu().numpy();rows[j]=dot.summarize(observations);completed+=n
            fingerprints.append(hashlib.sha256(observations.tobytes()).hexdigest())
            if j in audits:
                indices=np.unique(np.linspace(0,n-1,min(n,32),dtype=np.int64))
                raw={name:value[indices].cpu().numpy() for name,value in zip(
                    ['a','b','acc','p','q','hi','lo','ulp'],[*inputs,pv,qv,hi,lo,u])}
                observed,_=cpu_observations(*(raw[name] for name in ['a','b','acc','p','q','hi','lo','ulp']),exact=False)
                if not np.allclose(observed,observations[indices],rtol=2e-13,atol=1e-13):raise ValueError('CPU group observer differs')
                name=f'audit-{j:06d}.npz';np.savez_compressed(out/name,observations=observations,indices=indices,**raw)
                saved.append({'part':j,'file':name})
            now=time.monotonic()
            if now-last>=5 or j+1==len(counts):
                progress={'groups_completed':completed,'groups_total':p['settings']['samples'],
                          'parts_completed':j+1,'parts_total':len(counts),'elapsed_seconds':now-start}
                dot.common.write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True);last=now
        drawn=np.array(counts,np.int64)
        result=stats.evaluate_parts(rows,np.zeros(len(rows),np.int8),drawn,[1.],p['settings'],p['diagnostic'])
        np.savez_compressed(out/'statistics.npz',rows=rows,drawn=drawn)
        dot.common.write_json(out/'fingerprints.json',fingerprints);dot.common.write_json(out/'compiled.json',compiled)
        names=['statistics.npz','fingerprints.json','classifications.npz','compiled.json',*(a['file'] for a in saved)]
        record.update(state='COMPLETE',result=result,audits=saved,files={name:dot.file_hash(out/name) for name in names},
                      elapsed_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        dot.common.write_json(out/'record.json',record)
        report=replay(out);dot.common.write_json(out/'report.json',report);print(json.dumps(report,indent=2));return 0
    except BaseException as error:
        record.update(state='ERROR',reason=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-start)
        dot.common.write_json(out/'record.json',record);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['check','run']:
        p=sub.add_parser(command);p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int)
        if command=='run':p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('import');p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='check':print(json.dumps(plan(args.smoke,args.samples),indent=2));return 0
    if args.command=='run':return run(args)
    report=replay(args.bundle)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
