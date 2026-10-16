from pathlib import Path
import json,hashlib,shutil
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/triton-mul2-finite-acceptance-revision2-wip');O=Path('/tmp/veritile-standardization-20261015/triton-mul2-finite-acceptance-wip');V=Path('/tmp/veritile-standardization-20261015/triton-mul2-independent-review')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
old=load(O/'evidence/receipt.json');new=load(P/'evidence/receipt.json');lineage=new['revision_lineage']
assert sha(O/'evidence/receipt.json')==lineage['prior_receipt_sha256']==sha(P/lineage['prior_receipt'])
for rel,h in old['artifacts'].items():assert sha(O/rel)==h==sha(P/lineage['prior_artifacts'][rel]),rel
for rel,h in old['modules'].items():assert sha(O/'after'/rel)==h==sha(P/lineage['prior_modules'][rel]),rel
old_review=load(V/'review-receipt.json');assert sha(V/'review-receipt.json')=='c20026fa7c10d68c74cdf58998bcb027c4b338fc39a76972b12981d273864cda'
for rel,h in old_review['artifacts'].items():assert sha(V/rel)==h,rel
for rel,path in lineage['peer_evidence'].items():assert sha(V/rel)==sha(P/path)==old_review['artifacts'][rel]
# All original compiler inputs and artifacts are unchanged, so reuse the completed
# independent v1 offline compilation rather than rerunning unchanged 78 profiles.
for rr in new['native_evidence']['receipts']:
 assert sha(P/rr['receipt'])==rr['sha256']
 native=load(P/rr['receipt']);parent=(P/rr['receipt']).parent
 for rel,h in native['artifacts'].items():assert sha(parent/rel)==h,rel
for source in ['probe_pointer_execution.py','audit_grid_cases.py']:
 assert sha(W/'replay'/source)==sha(P/source)
for name in ['pointer-execution-probe.json','grid-case-audit.json']:
 assert load(W/'replay/evidence'/name)==load(P/'evidence'/name),name
compiles=load(P/'evidence/final-compiles.json');assert len(compiles['modules'])==9
for m in compiles['modules']:
 assert sha(P/m['source'])==m['source_sha256']
 assert sha(P/m['compiled_source'])==m['source_sha256']
 assert sha(P/m['object'])==m['object_sha256']
 assert sha(P/m['receipt'])==m['receipt_sha256']
 assert sha(P/m['log'])==m['log_sha256']
 receipt=load(P/m['receipt']);assert receipt['exit_code']==0
copydir=W/'reused-v1-evidence';copydir.mkdir(exist_ok=True)
for name in ['review-receipt.json','peer-review.json','source-review.md','replay-comparison.json','pointer-probe.json']:
 shutil.copy2(V/name,copydir/name)
report={'status':'passed','old_candidate_receipt_sha256':sha(O/'evidence/receipt.json'),'revision2_receipt_sha256':sha(P/'evidence/receipt.json'),'old_review_receipt_sha256':sha(V/'review-receipt.json'),'historical_artifacts_preserved':len(old['artifacts']),'historical_sources_preserved':len(old['modules']),'old_review_artifacts_unchanged':len(old_review['artifacts']),'copied_peer_artifacts_verified':len(lineage['peer_evidence']),'current_producer_source_object_log_receipt_sets_verified':len(compiles['modules']),'reused_native_profiles':78,'reused_equal_IR_artifacts':172,'new_pointer_wrapper_replay_cases':394,'new_grid_audit':{k:v for k,v in load(W/'replay/evidence/grid-case-audit.json').items() if k!='partitions'},'gpu_launches':0,'native_numerical_validation':False,'reused_v1_evidence':{name:sha(copydir/name) for name in ['review-receipt.json','peer-review.json','source-review.md','replay-comparison.json','pointer-probe.json']}}
(W/'reuse-and-replay-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['new_grid_audit','reused_v1_evidence']}))
