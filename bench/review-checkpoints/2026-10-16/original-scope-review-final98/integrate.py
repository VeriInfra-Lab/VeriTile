"""Record the remaining 98 substantive scope reviews without accepting tasks."""
import copy
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
from pathlib import Path

ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP=Path('/tmp/veritile-standardization-20261015')
BATCH=ROOT/'bench/review-checkpoints/2026-10-16/original-scope-review-final98'
MANIFEST=TMP/'scope98-review-manifest.json'
CHECK=TMP/'scope98-root-validation.json'
LEDGER=ROOT/'bench/original_program_scope.json'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import write_json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    assert not BATCH.exists(),'refuse checkpoint overwrite'
    assert not git('status','--porcelain'),'require clean integration base'
    assert git('branch','--show-current').decode().strip()=='fp-examples'
    base=git('rev-parse','HEAD').decode().strip()
    checked,manifest=read(CHECK),read(MANIFEST)
    assert checked['manifest_sha256']==sha(MANIFEST)
    assert checked['task_count']==98 and checked['whole_original_programs_accepted']==0
    assert checked['ledger_before_sha256']==sha(LEDGER)
    assert sha(ROOT/checked['catalog']['path'])==checked['catalog']['sha256']
    for path,h in checked['reviewed_input_sha256'].items():assert sha(ROOT/path)==h,path
    sources={row['destination']:row for row in manifest['reviews']+manifest.get('retained_artifacts',[])}
    assert len(sources)==len(manifest['reviews'])+len(manifest.get('retained_artifacts',[]))
    for row in sources.values():assert sha(Path(row['path']))==row['sha256']
    reviews={}
    for entry in manifest['reviews']:
        obj=read(Path(entry['path']))
        for row in obj.get('tasks',[obj]):reviews[row['task']]=(row,entry)
    assert set(reviews)==set(checked['tasks'])
    before=read(LEDGER);ledger=copy.deepcopy(before)
    assert len(ledger['tasks'])==183 and all(t['status']=='pending' for t in ledger['tasks'])
    names=set(reviews)
    previous={r['task']:r for r in before['tasks']}
    for row in ledger['tasks']:
        name=row['task']
        if name not in names:continue
        review,entry=reviews[name]
        assert row['initial_assessment']=='full_scope_not_yet_verified'
        row['initial_structural_gaps']=copy.deepcopy(row['structural_gaps'])
        row['structural_gaps']=copy.deepcopy(review['structural_gaps'])
        row['initial_assessment']='structural_gaps_confirmed'
        row['scope_gap_review']={'path':(BATCH/'evidence'/entry['destination']).relative_to(ROOT).as_posix(),
                                'sha256':entry['sha256'],'task':name,'status':'reviewed_gaps_not_acceptance'}
        assert row['review'] is None and not row['deferred_numerical_assumptions']
    assert all(r==previous[r['task']] for r in ledger['tasks'] if r['task'] not in names)
    assert sum(t['initial_assessment']=='structural_gaps_confirmed' for t in ledger['tasks'])==183
    ledger['scope_audit_checkpoint']=(BATCH/'README.md').relative_to(ROOT).as_posix()
    md=ROOT/'bench/original_program_scope.md';markdown=md.read_text()
    oldtext='The current full-scope review inventory is 85 programs with confirmed structural\ngaps and 98 without a completed scope review. All 183 remain pending.'
    newtext='The full-scope review now covers all 183 original programs. Each retains\nconcrete unresolved structural obligations, so all 183 remain pending. No\nprogram is left without a completed scope-gap review.'
    assert markdown.count(oldtext)==1
    # Exact validation of sources and intended row changes precedes every write.
    (BATCH/'evidence').mkdir(parents=True)
    shutil.copyfile(LEDGER,BATCH/'evidence/ledger-before.json')
    for dest,entry in sources.items():
        target=BATCH/'evidence'/dest;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(entry['path'],target);assert sha(target)==entry['sha256']
    for path,name in [(MANIFEST,'review-manifest.json'),(CHECK,'root-review.json'),
                      (TMP/'check_scope98_reviews.py','check-reviews.py'),(Path(__file__),'integrate.py')]:
        shutil.copyfile(path,BATCH/name)
    write_json(LEDGER,ledger)
    md.write_text(markdown.replace(oldtext,newtext))
    command=[sys.executable,'-B','bench/check_original_program_scope.py','--root',str(ROOT),
             '--catalog',str(ROOT/checked['catalog']['path']),'--ledger',str(LEDGER),'--output',str(BATCH/'scope-gate.json')]
    with (BATCH/'scope-gate.log').open('w') as output:
        process=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    gate=read(BATCH/'scope-gate.json')
    assert process.returncode==1 and gate['counts']=={'not_accepted':183}
    assert gate['errors']==['183 original-program scopes remain unaccepted']
    assert all(t['error']=='scope remains pending' for t in gate['tasks'])
    for path,h in checked['reviewed_input_sha256'].items():assert sha(ROOT/path)==h,path
    assert sha(ROOT/checked['catalog']['path'])==checked['catalog']['sha256']
    (BATCH/'README.md').write_text(f'''# Complete scope-gap inventory for the original 183 programs

All **183 original programs now have a substantive scope-gap review**.
Whole-program acceptance remains **0/183**, with **183 pending** and no
unreviewed programs. This batch adds reviews for the final 98 tasks, checking
{checked['public_declarations_checked']} public statements, {checked['structural_gap_count']} concrete gaps,
{checked['production_functions_checked']} production-function locations and
{checked['exact_excerpts_checked']} exact evidence ranges.

Existing complete kernels, typed profiles, loops, frame proofs and newer stage
proofs are explicitly credited in each review. Missing host composition is not
described as a missing kernel. Actual source quirks and refutations are retained;
the original programs have not been repaired or silently substituted.

All other 85 scope rows, every proof source, the current 942-declaration catalog
and numerical registrations are unchanged. Known historical declaration
location/hash discrepancies are recorded alongside exact current statements.
Batch04 group02 v2 supersedes its earlier layer-norm heuristic allegation;
the original record and correction remain as provenance, not current findings.

The scope checker exited 1 solely because all 183 scopes remain pending. No
Lean build or GPU/native run is claimed for this read-only review integration.
Compilation, review counts and numerical assumption registration do not count
as completed programs. Numerical conformance may be explicitly deferred only
after its exact statement/profile binding; genuine omitted stages, dtypes and
branches remain work.
''')
    paths={LEDGER.relative_to(ROOT).as_posix(),md.relative_to(ROOT).as_posix()}
    paths.update(p.relative_to(ROOT).as_posix() for p in BATCH.rglob('*') if p.is_file())
    validation={'schema':1,'kind':'remaining_original_scope_review_integration',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':base,
        'status':'complete_scope_gap_inventory_no_program_acceptance',
        'scope':{'requested':183,'accepted':0,'pending':183,'confirmed_gaps':183,'unreviewed':0},
        'new_scope_reviews':98,'new_public_statements_reviewed':checked['public_declarations_checked'],
        'new_concrete_structural_gaps':checked['structural_gap_count'],
        'checked_input_sha256':{p:sha(ROOT/p) for p in sorted(paths)},
        'reviewed_source_sha256':checked['reviewed_input_sha256'],
        'scope_gate_exit_code':process.returncode,'scope_gate_errors':gate['errors'],
        'preservation':{'other_scope_rows_unchanged':85,'public_statements_changed':0,
                        'lean_sources_changed':0,'original_python_changed':0,
                        'catalog_changed':False,'numerical_registrations_changed':False},
        'historical_metadata_discrepancies':checked['retained_catalog_metadata_discrepancies'],
        'acceptance_boundary':'Substantive source scope review plus exact evidence binding, without declaring any original program complete.'}
    write_json(BATCH/'validation.json',validation)
    print(json.dumps({'checkpoint':str(BATCH),'validation_sha256':sha(BATCH/'validation.json'),
                      'scope':validation['scope'],'new_publics_reviewed':checked['public_declarations_checked'],
                      'new_gaps':checked['structural_gap_count']},indent=2))

if __name__=='__main__':main()
