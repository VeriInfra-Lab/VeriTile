#!/usr/bin/env python3
"""Read-only lexical cross-check, not proof discovery or formal admission."""
from pathlib import Path
import collections, hashlib, json, re, sys
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path('/tmp/veritile-assumptions-20261014/design')
sys.path.insert(0,str(ROOT/'scripts'))
from spec_sheet import split_decls,split_statement,strip_comments
inventory=json.loads((ROOT/'bench/tritonbench_g/current_completion_inventory.json').read_text())
review=json.loads((ROOT/'bench/tritonbench_g/coverage_review.json').read_text())
rows=[]; files={}
for p in review['ports']:
    path=ROOT/p['file']; text=path.read_text(); files[p['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
    ds={d['name']:d for d in split_decls(text)}
    for e in p['declarations']:
        d=ds[e['declaration']]; s=strip_comments(split_statement(d['text'])[0])
        flags=[]
        if re.search(r'\bRules\b',s): flags.append('rules_container')
        if re.search(r'round[^\n]*=\s*id',s): flags.append('global_round_identity')
        if 'NativeAdditionCommutes' in s: flags.append('native_addition_commutes')
        if re.search(r'\(\w+\s*:\s*RoundingModel\)',s): flags.append('shared_rounding_model_parameter')
        if 'RoundingModel.triv' in s: flags.append('mathematical_triv_use_not_extra_law')
        if '≡[assumptions]' in s: flags.append('named_assumptions_reference_requires_resolution')
        if 'Spec.Assumptions FP.GuardedFragment)]' in s: flags.append('inline_empty_assumptions')
        if '≡[Forward.Floating.assumptions]' in s: flags.append('qualified_assumptions_reference_requires_resolution')
        rows.append(dict(task=path.parent.name,file=p['file'],declaration=e['declaration'],
            line=text[:d['start']].count('\n')+1,scope=e['coverage_level'],numeric_model=e['numeric_model'],
            flags=flags,statement=s))
counts={}
for flag in sorted({f for r in rows for f in r['flags']}):
    rs=[r for r in rows if flag in r['flags']]
    counts[flag]={'declarations':len(rs),'tasks':len({r['task'] for r in rs})}
cross=[]
for cat in sorted({t['scope_category'] for t in inventory['tasks']}):
    ts=[t for t in inventory['tasks'] if t['scope_category']==cat]
    ds=[d for t in ts for d in t['public_declarations']]
    cross.append(dict(category=cat,tasks=len(ts),declarations=len(ds),numeric_models=dict(collections.Counter(d['numeric_model'] for d in ds))))
partial=[]
for t in inventory['tasks']:
    if t['scope_category'] in ['partial_public_statement','blocked_public_statement','missing_source']:
        partial.append(dict(task=t['task'],old_category=t['scope_category'],
            scope_user_acceptance='no_source_to_review' if t['scope_category']=='missing_source' else 'accepted_for_this_review',
            full_source_blocked=t['task']=='quant_transpose_kernel',
            declarations=[{k:d[k] for k in ['file','declaration','coverage_level','numeric_model','evidence']} for d in t['public_declarations']]))
reports=[]
for relative in ['experiments/floating_point/report/summary.json','experiments/floating_point/supplement/report/summary.json','experiments/floating_point/primitives/report/summary.json','experiments/floating_point/supplement/log_report/summary.json','experiments/floating_point/supplement/exp_report/summary.json']:
    path=ROOT/relative; data=json.loads(path.read_text()); rs=data['rows']; reports.append(dict(file=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(rs),decisions=dict(collections.Counter(r['decision'] for r in rs)),caveat='Published report inventory only; no replay or new experiment executed.'))
for name,data in [('distribution.json',{'schema':1,'scope_numeric_cross_tab':cross,'lexical_review_flags':counts,'published_reports':reports}),('public_statement_crosscheck.json',{'schema':1,'method':'comment-stripped lexical public-statement scan; resolve alias/structures separately; not a proof-derived exhaustive classification','inputs':files,'rows':rows}),('partial_stage_review.json',{'schema':1,'rows':partial})]:
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(counts,ensure_ascii=False,indent=2))
