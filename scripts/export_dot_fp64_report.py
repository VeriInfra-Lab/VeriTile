"""Retain a replayed, precision-specific FP64 dot result without altering FP32 evidence."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path

if __package__:
    from . import two_gates_dot_fp64 as fp64
else:
    import two_gates_dot_fp64 as fp64

dot=fp64.dot
REPORT=fp64.DIRECTORY/'report/receipt.json'


def validate(receipt):
    m,r=receipt['manifest'],receipt['report']
    expected=fp64.plan(False,2**32)
    if (receipt['kind']!='dot_contraction_fp64_receipt' or m['format']!=fp64.FORMAT or
            any(m.get(k)!=v for k,v in expected.items()) or r.get('precision')!='fp64' or
            r['complete'] is not True or r['rule']!='DOT-ADD-TO-DOTACC' or
            r['manifest_sha256']!=dot.digest(m) or r['source_hashes']!=m['sources'] or r['backend']!=m['backend']):
        raise ValueError('FP64 receipt identity, precision, scope or fixed protocol differs')
    result=r['result'];s=result['sampling'];b=result['bias'];v=result['magnitude']
    if (s['drawn']!=2**32 or s['peak_drawn']!=2**32 or s['valid']!=s['peak_valid'] or
            not 2<=s['valid']<=s['in_domain']<=s['drawn'] or
            s['excluded_domain']!=s['drawn']-s['in_domain'] or s['excluded_nonfinite']!=s['in_domain']-s['valid'] or
            r['raw_audits_verified']!=len(dot.audit_indices(m)) or
            r['part_statistics_verified']!=len(dot.part_counts(m)) or
            r['exact_rational_output_audits']!=len(dot.audit_indices(m))*32*1024 or
            not 0<=r['max_audited_reference_error_ulp']<2.**-35):
        raise ValueError('FP64 fixed budget, independence or exact reference audit differs')
    strata=b['strata']
    if (len(strata)!=1 or strata[0]['probability']!=1. or strata[0]['n']!=s['valid'] or
            strata[0]['mean']!=b['mean'] or strata[0]['m2']<0 or
            not math.isclose(math.sqrt(strata[0]['m2']/(s['valid']-1)/s['valid']),
                             b['standard_error'],rel_tol=1e-14,abs_tol=1e-15)):
        raise ValueError('FP64 standard error differs from whole-group variance')
    bound=abs(b['mean'])+m['settings']['bias_se_multiplier']*b['standard_error']
    if (not all(math.isfinite(x) for x in [b['mean'],b['standard_error'],b['upper'],v['reference_peak'],v['candidate_peak']]) or
            b['standard_error']<0 or min(v['reference_peak'],v['candidate_peak'])<0 or
            not math.isclose(bound,b['upper'],rel_tol=1e-14,abs_tol=1e-15) or
            b['tolerance']!=m['settings']['bias_tolerance_ulp'] or v['limit']!=m['settings']['peak_ratio_limit']):
        raise ValueError('FP64 gate moments or thresholds differ')
    bp=bound<=b['tolerance'];vp=v['candidate_peak']<=v['limit']*v['reference_peak']
    decision='ACCEPT' if bp and vp else 'REJECT'
    ratio=0. if v['candidate_peak']==0 else v['candidate_peak']/v['reference_peak'] if v['reference_peak'] else '+inf'
    eligible=decision=='ACCEPT' and r['special']['classification_mismatches']==0 and r['special']['nan_payload_differences']==0
    if (result['decision']!=decision or b['status']!=('PASS' if bp else 'FAIL') or
            v['status']!=('PASS' if vp else 'FAIL') or v['ratio']!=ratio or
            r['all_words_admission_eligible']!=eligible):
        raise ValueError('FP64 recorded acceptance contradicts its own gates/scope')
    return eligible


def collect(bundle):
    report=fp64.replay(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    record=json.loads((bundle/'record.json').read_text())
    receipt={'kind':'dot_contraction_fp64_receipt','checked_at_utc':datetime.now(timezone.utc).isoformat(),
             'manifest':manifest,'report':report,'record_sha256':dot.file_hash(bundle/'record.json'),
             'measurement':{k:record[k] for k in ['elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes']},
             'trust':'Exact rational CPU raw audits and complete statistics replay; remote GPU execution remains trusted.'}
    validate(receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path);parser.add_argument('--receipt',type=Path,default=REPORT)
    parser.add_argument('--check',action='store_true');args=parser.parse_args(argv)
    if args.bundle and args.check:parser.error('--check uses the retained compact receipt')
    receipt=collect(args.bundle) if args.bundle else json.loads(args.receipt.read_text())
    eligible=validate(receipt)
    if args.bundle:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        with args.receipt.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'precision':'fp64','decision':receipt['report']['result']['decision'],
                      'eligible':eligible,'receipt':str(args.receipt),'checked':args.check}))
    return 0


if __name__=='__main__':raise SystemExit(main())
