"""Read-only structural binding to retained PTX; no compilation or numerical experiment."""
import ast
import hashlib
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'softmax_flaggems_finite_seed'
EVIDENCE = ROOT / '.veritile-cache/softmax-flaggems-finite-seed-v2-evidence'
REVIEW = ROOT / '.veritile-cache/independent-reviews/softmax_flaggems_finite_seed-resume_triton_attention'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_ptx(text):
    # Parse the actual instructions feeding the eight row-register stores.
    lines = [line.strip() for line in text.splitlines()]
    instructions = []
    pattern = re.compile(r'(mul\.f32|fma\.rn\.f32|add\.f32|sub\.f32|shfl\.sync\.bfly\.b32)\s+(.*);$')
    for line in lines:
        match = pattern.fullmatch(line)
        if match:
            instructions.append((match[1], [v.strip() for v in match[2].split(',')]))
    definitions = {args[0]:(op,args[1:]) for op,args in instructions}
    consumers = []
    for op,args in instructions:
        if op != 'sub.f32':
            continue
        sub, g, scale = args
        dest = [a for code,a in instructions if code=='mul.f32' and a[1]==sub]
        assert len(dest)==1
        out, _, p = dest[0]
        current = scale
        for distance in [1,2,4,8]:
            code, pair = definitions[current]
            assert code=='add.f32'
            own, shuffled = pair
            scode,sargs = definitions[shuffled]
            assert scode=='shfl.sync.bfly.b32' and sargs==[own,str(distance),'31','-1']
            current=own
        code,args = definitions[current]
        assert code=='fma.rn.f32' and args[:2]==[p,g]
        partner=args[2]
        scode,sargs=definitions[partner]
        assert scode=='shfl.sync.bfly.b32' and sargs[1:]==['16','31','-1']
        code,args=definitions[sargs[0]]
        assert code=='mul.f32' and args==[p,g]
        assert re.search(r'st\.global\.b32[^\n]*\{\s*'+re.escape(out)+r'\s*\}',text)
        consumers.append({'p':p,'g':g,'scale':scale,'store_word':out,'distances':[16,8,4,2,1]})
    assert len(consumers)==8
    assert '.param .u64 softmax_backward_kernel_inner_param_4' in text
    assert re.search(r'ld\.param\.b64\s+%rd\d+,\s*\[softmax_backward_kernel_inner_param_4\]',text)
    assert 'and.b32' in text and '31;' in text
    # The reduction is consumed in register. No shared publication/broadcast
    # occurs in this artifact, unlike the retained changed-forward profiles.
    assert not re.search(r'\b(?:ld|st)\.shared',text)
    return consumers


def main():
    source = HERE / 'softmax_flaggems_repaired.py'
    assert source.read_bytes()==(OLD/source.name).read_bytes()
    selected = HERE.parent/'softmax_flaggems'/source.name
    seed='value=-340282346638528859811704183484516925440.0, dtype=tl.float32'
    text=source.read_text()
    assert text.count(seed)==2
    restored=text.replace(seed,'value=float("-inf"), dtype=tl.float32')
    assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(selected.read_text()))
    retained=json.loads((ROOT/'.veritile-cache/softmax-flaggems-finite-seed-evidence/candidate-report.json').read_text())
    assert retained['source_sha256']==digest(source)
    assert retained['quote_sha256']==digest(OLD/'Kernels.lean')
    assert retained['source']['four_full_quotes_match']
    assert len(retained['source']['source_mutants_rejected'])==8 and len(retained['boundaries'])==40
    ptx_path=REVIEW/'supplemental-saved-evidence/softmax_backward_kernel_inner-N3-B32-C32.ptx'
    assert digest(ptx_path)=='7964499c77637c789de786c43553158a790bce746c1505eef5e6e54b9bdb40fb'
    ptx=ptx_path.read_text()
    consumers=check_ptx(ptx)
    # These mutate the retained instruction dataflow, not the Lean implementation.
    changes=[('first-butterfly-distance',', 16, 31, -1;',', 8, 31, -1;'),
             ('first-stage-fusion','fma.rn.f32','mad.rn.f32'),
             ('local-product-order','%r1, %r10, %r46','%r10, %r1, %r46'),
             ('consumer-scale','%r10, %r55','%r10, %r65'),
             ('final-multiply-order','%r126, %r1','%r1, %r126')]
    rejected=[]
    for label,a,b in changes:
        assert a in ptx,label
        try:
            check_ptx(ptx.replace(a,b,1))
        except (AssertionError,KeyError,ValueError):
            rejected.append(label)
        else:
            raise AssertionError('accepted profile mutant: '+label)
    review=json.loads((REVIEW/'review.json').read_text())
    assert review['decision']=='CHANGES_REQUIRED' and review['findings'][0]['id']=='F1'
    derivation=json.loads((REVIEW/'replica-counterexample.json').read_text())
    assert derivation['actual_stored_gradients']['lane_0'].endswith('(0x00000000)')
    assert derivation['actual_stored_gradients']['lane_16'].endswith('(0x80000000)')
    report={'candidate_source_sha256':digest(source),'v1_source_and_quote_unchanged':True,
            'python_changes_from_selected':'two seeds only','retained_full_quote_ast_checks':4,
            'retained_source_mutants':8,'retained_boundary_cases':40,
            'saved_ptx_sha256':digest(ptx_path),'saved_ptx_consumer_rows':consumers,
            'profile_mutants_rejected':rejected,
            'counterexample_evidence':'retained reviewer analytic derivation, not re-executed',
            'new_compilations':0,'new_numerical_experiments':0,'gpu_execution':False}
    (EVIDENCE/'profile-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source_unchanged':True,'ptx_consumers_checked':len(consumers),
                      'profile_mutants_rejected':len(rejected),'new_compilations':0,'new_numerical_experiments':0}))


if __name__=='__main__':
    main()
