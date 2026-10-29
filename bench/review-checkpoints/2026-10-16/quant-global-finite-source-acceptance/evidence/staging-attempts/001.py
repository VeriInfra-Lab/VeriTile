from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');B=R/'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance';dest=B/'staging-review.json'
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def sha(b):return hashlib.sha256(b).hexdigest()
assert not dest.exists();assert git('branch','--show-current').decode().strip()=='fp-examples'
v=json.loads((B/'validation.json').read_text());assert git('rev-parse','HEAD').decode().strip()==v['base_commit']
assert v['scope']=={'requested':183,'accepted':4,'verified':1,'verified_with_deferred_numerics':3,'pending':179,'reviewed':183,'unreviewed':0}
index=json.loads((B/'evidence-index.json').read_text())
expected={'.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md','bench/reviewed_numerical_extensions.py','bench/check_original_program_scope.py','bench/tests/test_exact_numerical_bindings.py','bench/tritonbench_g/deferred_numerical_experiments.json','bench/tritonbench_g/deferred_numerical_experiments.md'}|set(index['new_source_sha256'])|{p.relative_to(R).as_posix() for p in B.rglob('*') if p.is_file()}
staged={p for p in git('diff','--cached','--name-only','-z').decode().split('\0') if p};assert staged==expected,(staged-expected,expected-staged)
result=subprocess.run(['git','diff','--cached','--check'],cwd=R,capture_output=True);diagnostics=result.stdout.decode()+result.stderr.decode()
reported={line.split(':',1)[0] for line in diagnostics.splitlines() if ': trailing whitespace.' in line or ': new blank line at EOF.' in line}
allowed={(B/p).relative_to(R).as_posix() for p in index['retained_exact_artifacts']}|{(B/'evidence/integration/quant-global-finite-integration-build.log').relative_to(R).as_posix()}
assert reported<=allowed and result.returncode in [0,2],diagnostics
filtered=subprocess.run(['git','diff','--cached','--check','--','.',*[':(exclude)'+p for p in sorted(reported)]],cwd=R,capture_output=True);assert filtered.returncode==0,filtered.stdout.decode()+filtered.stderr.decode()
hashes={}
for name in sorted(staged):
 p=R/name;assert p.is_file() and not p.is_symlink();data=p.read_bytes();assert git('show',':'+name)==data,name;hashes[name]=sha(data)
for name,h in v['checked_input_sha256'].items():assert sha((R/name).read_bytes())==h,name
for p,h in index['retained_exact_artifacts'].items():assert sha((B/p).read_bytes())==h,p
record={'schema':1,'kind':'quant_global_finite_source_acceptance_staging_review','status':'passed','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':v['base_commit'],'staged_paths_before_this_record':len(staged),'staged_file_sha256':hashes,'validated_input_hashes_rechecked':len(v['checked_input_sha256']),'diff_cached_check_exit_code':result.returncode,'exact_evidence_whitespace':{'paths':sorted(reported),'diagnostics':diagnostics,'reason':'Frozen producer inputs and raw compiler diagnostics are retained byte-for-byte; only hash-verified evidence whitespace is excluded from source whitespace checks.','sha256':{p:hashes[p] for p in sorted(reported)}},'all_other_staged_files_diff_check_exit_code':filtered.returncode,'scope':v['scope'],'official_comparator_passed':False}
dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':'passed','staged_files':len(staged),'review_sha256':sha(dest.read_bytes())}))
