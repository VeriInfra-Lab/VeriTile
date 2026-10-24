"""Independent frozen-file audit; never writes package/repository files."""
import argparse,collections,datetime,hashlib,json,re,sys
from pathlib import Path
BASE=Path('/tmp/veritile-standardization-20261015')
ROOT=BASE/'relu-byte-memory-wip'
REPO=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=BASE/'relu-byte-memory-independent-review.json');args=parser.parse_args()
assert str(args.output).startswith('/tmp/') and not args.output.is_relative_to(ROOT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
receipt_sha='3623d3f83bb16d279d601ad8ce5a8a87c4e9565e84ba7be073cc76b165bec40b'
scope_sha='a1cac8990de499ec700642d1609405b1d76bbcd2d223e5d1dcb6e22fe3bdcee3'
assert sha(ROOT/'evidence/receipt.json')==receipt_sha
assert sha(ROOT/'evidence/scope-review.json')==scope_sha
r=read(ROOT/'evidence/receipt.json');scope=read(ROOT/'evidence/scope-review.json')
assert len(r['modules'])==4
for rel,h in r['modules'].items():
 p=ROOT/rel;assert sha(p)==h
 text=p.read_text();assert text.endswith('\n') and all(l==l.rstrip(' \t') for l in text.splitlines())
 assert not re.search(r'^\s*(?:axiom|sorry)\b',text,re.M)
for rel,h in r['artifacts'].items():assert sha(ROOT/rel)==h
before=read(ROOT/'evidence/final-dependencies-before.json');after=read(ROOT/'evidence/final-dependencies-after.json')
assert before==after
assert len(after['files'])==180
assert sum('olean_path' in row for row in after['files'].values())==172
source_hashes={};olean_hashes={}
for rel,row in after['files'].items():
 p=Path(row['source_path']);assert sha(p)==row['sha256'];source_hashes[str(p)]=row['sha256']
 if 'olean_path' in row:
  p=Path(row['olean_path']);assert sha(p)==row['olean_sha256'];olean_hashes[str(p)]=row['olean_sha256']
assert collections.Counter(v['source_kind'] for v in after['files'].values())=={'repository':167,'package':4,'relu-original-wip':3,'relu-integer-word-refinement-wip':3,'relu-finite-address-wip':3}
# Reconstruct all project import edges against the bound source resolution.
for rel,row in after['files'].items():
 if not rel.endswith('.lean'):continue
 for line in Path(row['source_path']).read_text().splitlines():
  if not line.startswith('import '):continue
  for mod in line.removeprefix('import ').split():
   if mod.startswith('--'):break
   dep=mod.replace('.','/')+'.lean'
   assert dep in after['files'] or dep in after['external_imports'],(rel,dep)
for directory,hashes in after['frozen_receipts'].items():
 p=Path(directory);assert sha(p/'evidence/receipt.json')==hashes['receipt']
 assert sha(p/'evidence/scope-review.json')==hashes['scope']
 for rel,h in read(p/'evidence/receipt.json')['modules'].items():assert sha(p/rel)==h
route=after['inherited_route_review'];assert sha(Path(route['path']))==route['sha256']
assert len(after['audited_ttir_sha256'])==13
for path,h in after['audited_ttir_sha256'].items():assert sha(Path(path))==h
results=read(ROOT/'evidence/verification-results.json');history=read(ROOT/'evidence/verification-attempts.json')
assert len(results)==4 and history[-4:]==results and len(history)==13
assert len([v for v in history if v['exit_code']!=0])==2
assert r['failed_attempts']==[v for v in history if v['exit_code']!=0]
for v in history:
 assert sha(ROOT/v['log'])==v['log_sha256']
 assert sha(ROOT/v['source_snapshot'])==v['snapshot_sha256']==v['source_sha256_before']==v['source_sha256_after']
 assert v['command']==['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(ROOT),str(ROOT/v['module']),'-o',str((ROOT/v['module']).with_suffix('.olean'))]
 assert v['seconds']>0
logs='\n'.join((ROOT/v['log']).read_text() for v in results)
assert [v['exit_code'] for v in results]==[0,0,0,0]
assert {v['module']:v['source_sha256_before'] for v in results}==r['modules']
assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b|warning:',logs)
assert logs.count('axiom footprint ⊆ standard base ✓')==30
assert logs.count('non-circular ✓')==3
texts={rel:(ROOT/rel).read_text() for rel in r['modules']}
assert sum(t.count('#axiomsClean ') for t in texts.values())==30
assert sum(t.count('#specNonCircular ') for t in texts.values())==3
assert sum(t.count('#auditModuleAxioms') for t in texts.values())==4
for text in texts.values():
 for name in re.findall(r'^#axiomsClean (\w+)',text,re.M):assert '.'+name+': axiom footprint' in logs
sys.path.insert(0,str(REPO/'bench'))
from adapter_common import declarations
public=r['public_contract'];decl=declarations(ROOT/public['file'])[public['parser_name']]
assert decl['kind']=='specification' and decl['statement_sha256']==public['statement_sha256']=='40bc0379ed4ffe9ab6d9e022f7edd1d78b476b420980114d94834264a0ec5818'
assert scope['public_contract']==public
assert public['unclosed_numeric_callbacks']==read(BASE/'relu-integer-word-refinement-wip/evidence/receipt.json')['public_contract']['unclosed_numeric_callbacks']
assert scope['new_numerical_callbacks']==[]
assert r['whole_original_program_accepted'] is False and r['original_program_completion_increment']==0
assert r['coverage_summary']['new_dtype_pairs']==0 and len(public['profiles'])==9
# Reviewed constraints must remain explicit in both proof source and scope.
public_text=texts[public['file']]
assert '(view : LogicalView' in public_text and '(distinct : x ≠ y)' in public_text and '(separate : Separate' in public_text
assert 'runGrid (backend M cmp16 cmp32) (source encoding.profile x y N blockSize)' in public_text
assert 'byteGrid M encoding inputBase outputBase N bytes = byteSpec' in public_text
limits='\n'.join(v['detail'] for v in r['acceptance_limits'])
assert 'memory-effect projection' in limits and 'does not imply their native load success' in limits
assert 'Native device bytes' in limits and 'nonoverlapping' in limits
report={
 'schema':1,'reviewer':'/root/full_scope_audit','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'accepted_partial_mathematical_byte_memory_refinement','proof_defects_found':[],
 'whole_original_program_accepted':False,'original_program_completion_increment':0,'new_dtype_profiles':0,
 'receipt_sha256':receipt_sha,'scope_review_sha256':scope_sha,
 'reviewed_source_sha256':r['modules'],'reviewed_input_sha256':source_hashes,
 'reviewed_imported_olean_sha256':olean_hashes,
 'reviewed_artifact_sha256':r['artifacts'],
 'public_statement_sha256':public['statement_sha256'],
 'checker_path':str(Path(__file__).resolve()),'checker_sha256':sha(Path(__file__).resolve()),
 'checks':{'artifact_fingerprints':len(r['artifacts']),'source_dependencies':180,'imported_project_oleans':172,'pre_post_dependency_snapshots_identical':True,'reconstructed_import_edges_complete':True,'inherited_raw_ttir_fingerprints':13,'actual_producer_lean_exits':[0,0,0,0],'independent_lean_rerun':False,'named_axiom_checks':30,'noncircularity_audits':3,'whole_module_axiom_audits':4,'warnings':0,'all13_compile_attempt_logs_and_source_snapshots_checked':True,'failed_attempts_retained':2,'public_declaration_parser_kind_and_statement_fingerprint_checked':True,'source_whitespace_checked':True,'git_staged_whitespace_rerun':False,'external_library_rebuild':False},
 'semantic_review':[
  {'module':'ByteCodec','finding':'loadLE recursively concatenates actual8-bit memory cells in little-endian order. load_bit locates each bit at base+bit/8 and bit%8; split/store and both roundtrips quantify arbitrary counts, words and complete memory maps. It is not a wide-cell relabeling.'},
  {'module':'ByteStores','finding':'Four-byte prefix stores satisfy an independent byte specification and complete frame. byteProgram computes pid0 outputs from pre-store memory and updates only min(N,1024) words. Every later program has identity memory effect. Empty N leaves memory unchanged in the mathematical grid.'},
  {'module':'ByteView','finding':'LogicalView binds complete tagged input cells and fp32 output cells for every index<N to actual byte loads. Distinct logical regions plus full input/output interval separation preserve input byte loads and establish final view, including arbitrary output-tail bytes. stateFromBytes proves satisfiability for arbitrary initial bytes without native allocation claims.'},
  {'module':'ByteGridCorrect','finding':'The public theorem derives actual old typed runGrid success independently from InputWords supplied by the initial view, then proves its final memory is represented by the new byteGrid. Conclusion also gives independent byteSpec, byte frame, finite-pointer input readback, all four active output bytes and inherited address certificate. No execution/result/refinement theorem is assumed as a premise.'}],
 'noncircularity_review':{'audited_definitions':['loadLE','byteSpec','LogicalView'],'public_premise_analysis':'Premises are representation, name distinction, numerical interval separation and address domains only; no runGrid success/result equation occurs as a hypothesis. Independent byteSpec uses initial bytes and wordValue via prefixSpec, not byteGrid/byteProgram/storePrefix. Inherited wordValue and wordSpec were independently audited in the frozen word package.'},
 'scope_limits':[
  'byteProgram is a memory-effect projection. Its nonzero-pid loaded/result functions can remain unevaluated; it cannot certify native load success or native faults on these paths.',
  'Total Nat-addressed byte memory and BitVec64-toNat readback are mathematical models; pointer provenance, allocation/access admission, native endianness/layout/alignment and hardware memory are not established.',
  'The relation is final-state representation, not stepwise original-Python/TTIR/compiler execution refinement. Masked/unused values, poison, traps, sanitizers and scheduling remain structural.',
  'Distinct logical names and contiguous separated intervals, B=1024, N<2^31 and representable endpoints are explicit selected-domain assumptions; no runtime guards or allocator freshness follow.',
  'Nine existing integer/bool profiles only. Integer-to-fp32 and signed-negative zero callbacks retain prior unverified numerical correspondence; no new callbacks or dtype profiles were added.'
 ],
 'acceptance_recommendation':'Integrate as the explicitly scoped mathematical byte-memory refinement while keeping all native/source/host structural gaps open. Whole-original-program acceptance remains0/183.'
}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(args.output),'report_sha256':sha(args.output),'checker_sha256':sha(Path(__file__).resolve()),'status':report['status'],'checks':report['checks']},indent=2))
