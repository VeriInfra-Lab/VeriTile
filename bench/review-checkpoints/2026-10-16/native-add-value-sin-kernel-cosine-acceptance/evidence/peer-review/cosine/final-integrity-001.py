from pathlib import Path
import hashlib,json,re
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/cosine-native-finite-acceptance-revision1-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
r=load(P/'evidence/receipt.json');pin=(W/'producer-pin.txt').read_text().strip();assert sha(P/'evidence/receipt.json')==pin
for rel,h in r['artifacts'].items():assert sha(P/rel)==h,rel
for rel,h in r['modules'].items():
 assert sha(P/'after'/rel)==h,rel
 assert sha(P/r['checkpoint_module_source_paths'][rel])==h,rel
 assert sha(W/'isolated'/rel)==h,rel
for rel,h in r['module_objects'].items():assert sha(W/'isolated'/rel)==h,rel
b=load(W/'dependencies-before.json');a=load(W/'dependencies-after.json');assert a==b
old=load(P/'evidence/dependencies-before.json');ext=load(P/'evidence/dependencies-final-before.json')
for mod,d in b['project'].items():
 rel=mod.replace('.','/')+'.lean';entry=old['project'][rel]
 assert sha(Path(d['source']))==d['sha256']==entry['sha256'],mod
 assert sha(Path(d['olean']))==d['olean_sha256']==entry['object_sha256'],mod
for mod,d in b['external_objects'].items():
 rel=mod.replace('.','/')+'.olean'
 assert sha(Path(d['olean']))==d['sha256']==ext['external_objects'][rel]['object_sha256'],mod
logs=[(W/x['log']).read_text() for x in load(W/'independent-build-receipt.json')['modules']]
assert all('warning:' not in t and 'error:' not in t for t in logs)
sources=[(W/'isolated'/rel).read_text() for rel in r['modules']]
counts={tag:sum(len(re.findall(r'^'+re.escape(tag)+r'\b',t,re.M)) for t in sources) for tag in ['#axiomsClean','#specNonCircular','#auditModuleAxioms']}
assert counts=={'#axiomsClean':48,'#specNonCircular':7,'#auditModuleAxioms':9},counts
assert sum(t.count('axiom footprint') for t in logs)==48
reg=(W/'regression-001.log').read_text();assert load(W/'regression-001.json')['exit_code']==0 and reg.count('axiom footprint')==12
for text in sources:assert not re.search(r'\b(sorry|admit|native_decide)\b',text)
retained={str(p.relative_to(R)):sha(p) for p in (R/'bench/tritonbench_g/cosine_compute').rglob('*') if p.is_file() and p.suffix in ['.lean','.py','.md']}
assert not set(retained).intersection(r['modules'])
assert not set(retained).intersection(str(p.relative_to(P/'after')) for p in (P/'after').rglob('*') if p.is_file())
(W/'retained-repository-files.json').write_text(json.dumps({'scope':'current old-source snapshot; producer candidate is additive and contains no old file replacement; this is not a historical Git comparison','files':retained},indent=2)+'\n')
out={'status':'passed','producer_receipt_sha256':pin,'frozen_artifacts_verified':len(r['artifacts']),'canonical_modules_verified':len(r['modules']),'independent_objects_equal_producer':len(r['module_objects']),'project_source_and_objects_verified':len(b['project']),'external_objects_equal_producer':len(b['external_objects']),'dependencies_unchanged':True,'dependency_capture_timing':{'producer':old['capture_phase'],'reviewer':'before first independent build/import'},'final_warnings':0,'source_audits':counts,'independent_regression_axiom_checks':12,'retained_current_source_files':len(retained),'replacement_overlap':0}
(W/'final-integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
