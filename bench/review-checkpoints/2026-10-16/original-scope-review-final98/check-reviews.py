"""Read-only binding and evidence checks for the remaining original scopes."""
import ast
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path

ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP=Path('/tmp/veritile-standardization-20261015')
MANIFEST=TMP/'scope98-review-manifest.json'
DEST=TMP/'scope98-root-validation.json'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import canonical,declarations,sha,split_statement

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    assert not DEST.exists(),'refuse frozen validation overwrite'
    manifest=read(MANIFEST)
    ledger=read(ROOT/'bench/original_program_scope.json')
    catpath=ROOT/'bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax/current-source-catalog.json'
    catalog=read(catpath)
    assert ledger['catalog_sha256']==sha(canonical(catalog))
    rows={r['task']:r for r in ledger['tasks']}
    cats={r['task']:r for r in catalog['tasks']}
    expected={n for n,r in rows.items() if r['initial_assessment']=='full_scope_not_yet_verified'}
    assert len(expected)==98
    source_hashes={};reviews={};read_files={};parsed={};asts={};excerpts=0;functions=0;publics=0;gaps=0
    discrepancies=[]
    for file in manifest['reviews']:
        p=Path(file['path']);assert digest(p)==file['sha256'],p
        obj=read(p)
        cohort=obj.get('tasks',[obj])
        if 'tasks' in obj:
            assert obj['task_count']==len(cohort)
            assert obj['public_declarations_checked']==sum(len(t['public_declarations_checked']) for t in cohort)
            assert obj['structural_gap_count']==sum(len(t['structural_gaps']) for t in cohort)
        for review in cohort:
            name=review['task'];assert name in expected and name not in reviews,name
            assert review['status']=='structural_gaps_confirmed'
            assert review['whole_original_program_accepted'] is False and review['completion_increment']==0
            assert review['all_input_sha256']==rows[name]['input_sha256'],name
            for src,h in review['source_sha256'].items():
                assert digest(ROOT/src)==h,src
                read_files[src]=h
            for mapping in review['all_input_sha256'].values():
                for src,h in mapping.items():
                    assert digest(ROOT/src)==h,src
                    source_hashes[src]=h
            oldcat=review['catalog'];assert digest(ROOT/oldcat['path'])==oldcat['sha256']
            observed={(d['file'],d['declaration']):d['statement_sha256'] for d in review['public_declarations_checked']}
            wanted={(d['file'],d['declaration']):d['statement_sha256'] for d in cats[name]['declarations']}
            assert observed==wanted,name
            for d in review['public_declarations_checked']:
                src=d['file']
                if src not in parsed:
                    path=ROOT/src;parsed[src]=(path.read_text(),declarations(path),digest(path))
                text,ds,h=parsed[src];actual=ds[d.get('parser_name',d['declaration'])]
                assert actual['statement_sha256']==d['statement_sha256']
                assert actual['kind']=='specification'
                if 'sha256' in d:assert d['sha256']==h
                if 'line' in d:assert d['line']==text.count('\n',0,actual['start'])+1
                if 'statement' in d:assert split_statement(actual['text'])[0]==d['statement']
                publics+=1
            for stage in review['production_stage_mapping']:
                src=stage['path'];assert digest(ROOT/src)==stage['sha256']
                if src not in asts:asts[src]=ast.parse((ROOT/src).read_text())
                matches=[n for n in ast.walk(asts[src]) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                         and n.name==stage['function'] and n.lineno==stage['line'] and n.end_lineno==stage['end_line']]
                assert len(matches)==1,(name,stage)
                functions+=1
            assert review['structural_gaps'],name
            for gap in review['structural_gaps']:
                assert gap['can_defer_as_numeric_assumption'] is False
                assert gap.get('evidence'),(name,gap)
                for e in gap['evidence']:
                    path=ROOT/e['path'];assert digest(path)==e['sha256']
                    lines=path.read_text().splitlines();end=e.get('end_line',e['line'])
                    assert 1<=e['line']<=end<=len(lines)
                    if 'excerpt' in e:assert '\n'.join(lines[e['line']-1:end])==e['excerpt']
                    excerpts+=1
                gaps+=1
            discrepancies.extend({'task':name,'discrepancy':d} for d in review.get('catalog_declaration_metadata_discrepancies',[]))
            reviews[name]={'path':file['path'],'sha256':file['sha256'],'destination':file['destination'],
                           'public_count':len(observed),'gap_count':len(review['structural_gaps'])}
    assert set(reviews)==expected,(set(reviews)-expected,expected-set(reviews))
    for f in manifest.get('retained_artifacts',[]):assert digest(Path(f['path']))==f['sha256']
    result={'schema':1,'kind':'root_remaining_scope_review_validation','reviewer':'root',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'scope_gaps_reviewed_no_acceptance','manifest_sha256':digest(MANIFEST),
        'ledger_before_sha256':digest(ROOT/'bench/original_program_scope.json'),
        'catalog':{'path':catpath.relative_to(ROOT).as_posix(),'sha256':digest(catpath)},
        'task_count':len(reviews),'public_declarations_checked':publics,'structural_gap_count':gaps,
        'production_functions_checked':functions,'exact_excerpts_checked':excerpts,
        'reviewed_input_sha256':source_hashes,'substantively_read_files':read_files,
        'tasks':reviews,'retained_catalog_metadata_discrepancies':discrepancies,
        'root_direct_reviews':manifest['root_direct_reviews'],
        'independent_root_source_spot_checks':manifest['independent_root_source_spot_checks'],
        'review_boundary':'All task review conclusions and preserved coverage were read. Independent source spot checks are listed, not claimed for every line. Exact statements, task-input sets, stage AST coordinates and quoted evidence were mechanically checked for every reviewed task.',
        'whole_original_programs_accepted':0,'new_numerical_registrations':0,'proof_sources_changed':0,
        'other_scope_rows_to_preserve':85}
    DEST.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'validation':str(DEST),'sha256':digest(DEST),'tasks':len(reviews),'publics':publics,'gaps':gaps,'excerpts':excerpts}))

if __name__=='__main__':main()
