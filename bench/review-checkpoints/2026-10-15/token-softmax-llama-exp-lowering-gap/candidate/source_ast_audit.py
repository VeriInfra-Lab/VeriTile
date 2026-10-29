from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;B=Path('/tmp/veritile-token-softmax-bloom-word-20261015');W=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected='5a44de64505a9d9dd2df9def1d9484ca63de11e4c2df9a533696cffb1d73d08e';assert sha(B/'receipt.json')==expected
br=json.loads((B/'receipt.json').read_text());bp=B/'native/original.py';lp=P/'native/original.py'
assert sha(bp)==br['artifacts_sha256']['native/original.py']
assert sha(lp)==sha(W/'bench/tritonbench_g/token_softmax_llama/token_softmax_llama.py')
rows=[]
for name in ['_fwd_kernel_token_softmax','token_softmax_fwd']:
 def extract(path):
  text=path.read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(nodes)==1;n=nodes[0]
  first=min([n.lineno]+[d.lineno for d in n.decorator_list]);segment=''.join(text.splitlines(keepends=True)[first-1:n.end_lineno])
  return {'node':n,'segment':segment,'first':first,'last':n.end_lineno,'AST':ast.dump(n,include_attributes=False)}
 b,l=extract(bp),extract(lp);assert b['AST']==l['AST'];same=b['segment']==l['segment']
 rows.append({'function':name,'AST_equal':True,'full_decorated_segment_bytes_equal':same,'canonical_AST_sha256':hashlib.sha256(l['AST'].encode()).hexdigest(),'bloom_lines':[b['first'],b['last']],'llama_lines':[l['first'],l['last']],'bloom_segment_sha256':hashlib.sha256(b['segment'].encode()).hexdigest(),'llama_segment_sha256':hashlib.sha256(l['segment'].encode()).hexdigest()})
 (P/(name+'-canonical-ast.txt')).write_text(l['AST']+'\n')
r={'schema':1,'bloom_parent_receipt_sha256':expected,'bloom_original_sha256':sha(bp),'llama_original_sha256':sha(lp),'functions':rows,'unchanged_shared_contracts':{'llama_FPCorrect_sha256':sha(W/'bench/tritonbench_g/token_softmax_llama/FPCorrect.lean'),'llama_TypedShape_sha256':sha(W/'bench/tritonbench_g/token_softmax_llama/Proofs/TypedShape.lean'),'shared_TypedContract_sha256':sha(W/'bench/tritonbench_g/token_softmax_bloom/Proofs/TypedContract.lean')},'not_claimed':'Equality of whole Python modules (their test data and dtype differ); equality of source and GPU outputs; a new numerical proof. The exact kernel and host bodies are equal; Llama default test dtype is fp32.'}
(P/'source-ast-audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('PASS kernel+host AST',[(r['function'],r['full_decorated_segment_bytes_equal']) for r in rows])
