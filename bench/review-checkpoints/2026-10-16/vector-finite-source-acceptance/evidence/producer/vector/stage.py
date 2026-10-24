from pathlib import Path
import json
W=Path(__file__).resolve().parent
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
names=['Views','Memory','Routes','Schedule','Indices','FullScope']
mapping={n:('bench/tritonbench_g/vector_addition/FiniteCorrect.lean' if n=='FullScope' else 'bench/tritonbench_g/vector_addition/Proofs/Finite'+n+'.lean') for n in names}
for name,path in mapping.items():
    s=(W/'VectorFinite'/f'{name}.lean').read_text()
    for n,target in mapping.items():s=s.replace('import VectorFinite.'+n+'\n','import '+target[:-5].replace('/','.')+'\n')
    p=W/'after'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
(W/'evidence/staging-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
s=(W/'compile.py').read_text().replace("env['LEAN_PATH']=str(WORK)+':'+base","env['LEAN_PATH']=str(WORK/'build-overlay')+':'+base").replace('path=WORK/sys.argv[1]',"path=WORK/'after'/sys.argv[1]").replace("'--root='+str(WORK)","'--root='+str(WORK/'after')").replace("f'{seq:03d}-{path.stem}'","f'{seq:03d}-staged-{path.stem}'").replace("'module':sys.argv[1]","'module':'after/'+sys.argv[1]").replace("str(path.with_suffix('.olean'))","str((WORK/'build-overlay'/sys.argv[1]).with_suffix('.olean'))")
(W/'compile_staged.py').write_text(s)
for p in (R/'.lake/build/lib/lean/bench').rglob('*'):
    if p.is_file():
        q=W/'build-overlay'/p.relative_to(R/'.lake/build/lib/lean')
        q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():q.symlink_to(p)
for rel in mapping.values():(W/'build-overlay'/rel).parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(mapping))
