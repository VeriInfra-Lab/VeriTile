"""Check actual def-use paths in frozen original-kernel IR/PTX, not opcode counts."""
from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent
N=P/'native'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
REG=re.compile(r'%[A-Za-z][A-Za-z0-9_]*')
KNOWN_WRITES={'add','sub','mul','mad','fma','max','min','mov','cvt','cvta','shfl','ld','setp','set','and','or','xor','not','selp','div','ex2','shr','shl','bfe','bfi','lop3','neg','abs','prmt','isspacep','activemask','vote','redux'}
KNOWN_NO_WRITE={'st','red','bar','membar','fence','bra','ret','exit','trap','call'}
def events(ptx):
 targets=set()
 for line in ptx.splitlines():
  if re.search(r'\b(?:bra|call|brx|jmp)(?:\.\w+)*\s',line):targets.update(re.findall(r'\$L__[A-Za-z0-9_]+',line))
 result=[]
 for lineno,line in enumerate(ptx.splitlines(),1):
  t=line.strip()
  if not t or t.startswith('//') or t.startswith('.') or t in ('{','}'):continue
  if t.endswith(':'):
   if not re.fullmatch(r'\$L__tmp\d+:',t) or t[:-1] in targets:result.append({'line':lineno,'text':t,'control':True,'dests':[],'args':[],'op':'label','predicated':False,'unknown':False})
   continue
  m=re.match(r'(?:(@!?%\w+)\s+)?([A-Za-z]\w*(?:\.\w+)*)\s*(.*);$',t)
  if not m:
   if REG.search(t):result.append({'line':lineno,'text':t,'control':False,'dests':REG.findall(t),'args':[],'op':'UNKNOWN','predicated':False,'unknown':True})
   continue
  pred,op,operands=m.groups();base=op.split('.')[0]
  first=operands.split('}',1)[0]+'}' if operands.startswith('{') else operands.split(',',1)[0]
  dests=[] if base in KNOWN_NO_WRITE else REG.findall(first)
  args=[x.strip() for x in operands.split(',')[1:]]
  result.append({'line':lineno,'text':t,'op':op,'dst':dests[0] if len(dests)==1 else None,'dests':dests,'args':args,'predicated':bool(pred),'unknown':base not in KNOWN_WRITES|KNOWN_NO_WRITE,'control':base in {'bra','brx','jmp','call','ret','exit','trap','bar','membar','fence'}})
 return result

def span(events,producer,consumer,reg,strict):
 middle=[e for e in events if producer['line']<e['line']<consumer['line']]
 for e in middle:
  need(reg not in e['dests'],'register overwritten between producer and consumer: '+reg)
  need(not(e['unknown'] and reg in REG.findall(e['text'])),'unknown possible register write: '+reg)
 controls=[e for e in middle if e['control'] or e['unknown']]
 if strict:need(not controls,'control boundary in supposedly straight scalar chain')
 return {'status':'CONTROL_OR_SYNCHRONIZATION_NOT_PROVED' if controls else 'STRAIGHT_REGISTER_SPAN',
   'producer_line':producer['line'],'consumer_line':consumer['line'],'register':reg,'controls':controls}

def inspect(ptx,ttir):
 ev=events(ptx);definitions={};exps=[];divs=[]
 for row in ev:
  op=row['op'];dst=row.get('dst');args=row['args']
  if op.startswith('ex2.'):
   need(op=='ex2.approx.f32' and not row['predicated'],'unknown/predicated exponent or FTZ profile')
   mul=definitions.get(args[0]);need(mul and mul['op'] in ('mul.rn.f32','mul.f32') and not mul['predicated'],'ex2 not directly fed by unconditional RN/default-RN multiply')
   need('0f3FB8AA3B' in mul['args'],'wrong log2e encoding')
   source=next(a for a in mul['args'] if a!='0f3FB8AA3B');sub=definitions.get(source)
   need(sub and sub['op'] in ('sub.rn.f32','sub.f32') and not sub['predicated'],'scaled input not unconditional row-minus-max subtract')
   spans=[span(ev,sub,mul,source,True),span(ev,mul,row,args[0],True)]
   exps.append({'subtract':sub,'scale_multiply':mul,'ex2':row,'checked_straight_spans':spans})
  if op.startswith('div.'):
   need(op=='div.full.f32' and not row['predicated'],'division profile changed/predicated')
   current=definitions.get(args[0]);exp=next((e for e in exps if e['ex2'] is current),None)
   need(exp is not None,'numerator is not latest unmodified ex2 output')
   divs.append({'div':row,'numerator_ex2_line':exp['ex2']['line'],'denominator_register':args[1],
     'numerator_span':span(ev,exp['ex2'],row,args[0],False)})
  for name in row['dests']:definitions[name]=row
 need(exps and len(exps)==len(divs),'unmatched exp/div sites')
 # Full high-level SSA chain, including the reduction combiner and stored result.
 patt=r'(%[\w]+) = arith.subf (%[\w]+), (%[\w]+) : tensor<\d+xf32>'
 sub=re.search(patt,ttir);need(sub is not None,'missing fp32 source subtract')
 out,row,maxsplat=sub.groups()
 maximum=re.search(re.escape(maxsplat)+r' = tt.splat (%[\w]+) : f32 ->',ttir);need(maximum,'missing max broadcast')
 maxname=maximum[1]
 maxreduce=re.search(re.escape(maxname)+r' = "tt.reduce"\('+re.escape(row)+r'\).*?arith.maxnumf.*?tt.reduce.return',ttir,re.S)
 need(maxreduce,'maxnum reduction must consume the loaded fp32 row')
 exp=re.search(r'(%[\w]+) = math.exp '+re.escape(out)+r' :',ttir);need(exp,'source exp input differs');numerator=exp[1]
 sumred=re.search(r'(%[\w]+) = "tt.reduce"\('+re.escape(numerator)+r'\).*?arith.addf.*?tt.reduce.return',ttir,re.S);need(sumred,'sum must consume same numerators');den=sumred[1]
 dsplat=re.search(r'(%[\w]+) = tt.splat '+re.escape(den)+r' : f32 ->',ttir);need(dsplat,'missing denominator broadcast')
 divide=re.search(r'(%[\w]+) = arith.divf '+re.escape(numerator)+r', '+re.escape(dsplat[1])+r' :',ttir);need(divide,'divide operands not exp/sum')
 final=divide[1];cast=re.search(r'(%[\w]+) = arith.truncf '+re.escape(final)+r' :',ttir)
 if cast:final=cast[1]
 store=re.search(r'tt.store (%[\w]+), '+re.escape(final)+r', (%[\w]+) :',ttir);need(store,'div/cast result not masked store payload')
 load=re.search(r'(%[\w]+) = tt.load (%[\w]+), '+re.escape(store[2])+r', (%[\w]+) :',ttir);need(load,'input/output masks differ')
 # Widen for half/bfloat, direct loaded value for fp32.
 need(row==load[1] or re.search(re.escape(row)+r' = arith.extf '+re.escape(load[1])+r' :',ttir),'row not actual load or widening')
 return {'source_ssa':{'load':load[1],'mask':store[2],'maximum':maxname,'subtract':out,'exp':numerator,'sum':den,'divide':divide[1],'store_payload':final},'ptx_exponent_chains':exps,'ptx_division_chains':divs,'limitations':'Finite textual instruction-span checks only. Unknown/predicated writes and scalar-chain control boundaries reject; div spans crossing control/synchronization are explicitly not proved. Debug-only $L__tmp labels are ignored. PTX lane/reduction layout, CFG paths, memory/synchronization and machine instruction implementation are not proved; this is not a general PTX parser/interpreter.'}
def main():
 receipt=json.loads((N/'receipt.json').read_text());rows=[]
 for r in receipt['records']:
  for rel,h in r['artifact_sha256'].items():need(sha(N/rel)==h,'artifact drift '+rel)
  folder=N/r['id'];result=inspect((folder/'kernel.ptx').read_text(),(folder/'kernel.ttir').read_text())
  rows.append({'id':r['id'],'ptx_sha256':sha(folder/'kernel.ptx'),'ttir_sha256':sha(folder/'kernel.ttir'),**result})
 (P/'native-dataflow.json').write_text(json.dumps({'schema':1,'profiles':rows,'profile_count':len(rows),'native_execution':False,'native_receipt_sha256':sha(N/'receipt.json')},indent=2)+'\n')
 print('PASS',len(rows),'profiles',sum(len(r['ptx_exponent_chains']) for r in rows),'exact exp/div sites')
if __name__=='__main__':main()
