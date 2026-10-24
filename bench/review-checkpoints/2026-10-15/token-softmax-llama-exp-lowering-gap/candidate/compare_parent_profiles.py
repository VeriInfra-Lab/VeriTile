from pathlib import Path
import json,hashlib,re
P=Path(__file__).resolve().parent;B=Path('/tmp/veritile-token-softmax-bloom-word-20261015')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def executable_text(path):
 result=[]
 for line in path.read_text().splitlines():
  s=line.split('//',1)[0].strip()
  if s.startswith('.file') or s.startswith('.section .debug'):break
  if not s or s.startswith('.loc') or re.fullmatch(r'\$L__tmp\d+:',s):continue
  result.append(' '.join(s.split()))
 return '\n'.join(result)+'\n'
ours=json.loads((P/'native/receipt.json').read_text())['records'];parent={r['id']:r for r in json.loads((B/'native/receipt.json').read_text())['records']};rows=[]
for r in ours:
 if r['id'] not in parent:continue
 a=P/'native'/r['id']/'kernel.ptx';b=B/'native'/r['id']/'kernel.ptx'
 assert sha(b)==parent[r['id']]['artifact_sha256'][r['id']+'/kernel.ptx']
 x,y=executable_text(a),executable_text(b)
 rows.append({'profile':r['id'],'llama_ptx_sha256':sha(a),'bloom_ptx_sha256':sha(b),'instructions_equal_after_debug_removal':x==y,'llama_normalized_sha256':hashlib.sha256(x.encode()).hexdigest(),'bloom_normalized_sha256':hashlib.sha256(y.encode()).hexdigest()})
assert len(rows)==12
out={'rows':rows,'compared_profiles':len(rows),'equal_profiles':sum(x['instructions_equal_after_debug_removal'] for x in rows),'normalization':'Whitespace/comments, .loc/debug-only $L__tmp labels removed; stop at trailing .file/debug section. Keep all executable instructions, nondebug control labels, registers, constants, parameters, target/version. This is a textual comparison, not a PTX/native semantic proof.'}
(P/'parent-profile-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print('PTX compare',out['equal_profiles'],'/',len(rows))
