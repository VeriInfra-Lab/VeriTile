from pathlib import Path
from collections import Counter
import json,hashlib,re,difflib
T=Path('/tmp/veritile-standardization-20261015');P=T/'add-value-finite-acceptance-wip';V=Path(__file__).resolve().parent
prod=next((P/'native-attempts').glob('*/receipt.json'));rep=next((V/'replay/native-attempts').glob('*/receipt.json'))
a=json.loads(prod.read_text());b=json.loads(rep.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for field in ['source_sha256','production_ast_unchanged','torch_version','torch_commit','triton_version','gpu_launches']:assert a[field]==b[field],field
counts=Counter();ir=0;debug_differences=[]
for x,y in zip(a['records'],b['records'],strict=True):
 assert x['dtype']==y['dtype'] and x['exit_code']==y['exit_code']==0
 x=x['result'];y=y['result'];assert x['status']==y['status'];counts[x['status']]+=1
 for field in ['dtype','target','value_python','N_python','signature','specialization','constexprs','attrs','options','parameters','value_word_fp32','failure_stage']:
  assert x.get(field)==y.get(field),(x['dtype'],field)
 if x['status']=='compiled':
  for ext in ['raw.ttir','ttir','ttgir','llir','ptx']:
   name=x['dtype']+'.'+ext;s=(prod.parent/name).read_text().replace(str(prod.parent),'<work>');t=(rep.parent/name).read_text().replace(str(rep.parent),'<work>')
   if ext=='ptx' and s!=t:
    left=re.split(r'\.section\s+\.debug_',s,maxsplit=1);right=re.split(r'\.section\s+\.debug_',t,maxsplit=1)
    assert len(left)==len(right)==2 and not re.search(r'\.(?:entry|func)\b',left[1]+right[1])
    assert left[0]==right[0],name
    # Debug sections encode the absolute source path as byte values, including
    # path-dependent DWARF unit lengths. Retain the complete difference.
    diff=''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile='producer/'+name,tofile='replay/'+name))
    dst=V/'replay-debug-diffs'/name;dst.parent.mkdir(exist_ok=True);dst.write_text(diff)
    debug_differences.append({'artifact':name,'diff_path':str(dst.relative_to(V)),'diff_sha256':sha(dst)})
   else:assert s==t,(name,'normalized IR differs')
   ir+=1
for receipt,data in [(prod,a),(rep,b)]:
 for name,h in data['artifacts'].items():assert sha(receipt.parent/name)==h,name
assert counts=={'compiled':15,'rejected':7}
x=json.loads((P/'evidence/view-and-schedule-probe.json').read_text());y=json.loads((V/'replay/evidence/view-and-schedule-probe.json').read_text())
for d in [x,y]:
 for m in d['torch_cpu_metadata']:
  for field in ['input_data_ptr','input_storage_base','output_data_ptr','output_storage_base']:m[field]=bool(m[field])
assert x==y
out={'status':'passed','producer_receipt_sha256':sha(prod),'replay_receipt_sha256':sha(rep),'native_profiles':22,'compiled':15,'rejected':7,'normalized_ir_or_executable_ptx_artifacts_identical':ir,'retained_debug_path_differences':debug_differences,'host_and_metadata_counts':x['counts'],'gpu_launches':0,'earlier_comparison_failure':'A preliminary raw PTX equality check stopped at bool.ptx because absolute source paths are encoded as DWARF bytes. This check compares identical executable PTX and retains every debug-section difference. No source or generated artifact was edited.'}
(V/'replay-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='retained_debug_path_differences'},indent=2))
