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
  t=line.split('//',1)[0].strip()
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

def inspect(ptx,ttir,scaled,fusion):
 ev=events(ptx);definitions={};exps=[]
 def current(reg,why):
  r=definitions.get(reg);need(r is not None,why);return r
 for row in ev:
  if row['op'].startswith('ex2.'):
   need(row['op']=='ex2.approx.f32' and not row['predicated'],'unknown/predicated exponent or FTZ profile')
   mul=current(row['args'][0],'missing exp argument multiply')
   need(mul['op'] in ('mul.rn.f32','mul.f32') and not mul['predicated'],'exp argument is not RN multiply')
   need(len(mul['args'])==2 and '0f3FB8AA3B' in mul['args'],'wrong log2e word')
   arg=next(a for a in mul['args'] if a!='0f3FB8AA3B');pre=current(arg,'missing exponent prefix')
   need(not pre['predicated'],'predicated prefix')
   expect='SCALED_FMA' if scaled and fusion else 'SEPARATE_SUB'
   paths=[span(ev,pre,mul,arg,True),span(ev,mul,row,row['args'][0],True)]
   if expect=='SCALED_FMA':
    need(pre['op']=='fma.rn.f32' and len(pre['args'])==3,'expected actual scale/sub contraction')
    neg=current(pre['args'][2],'missing negated maximum')
    need(neg['op']=='neg.f32' and not neg['predicated'],'FMA addend is not actual negation')
    paths.append(span(ev,neg,pre,pre['args'][2],True));maxreg=neg['args'][0]
    candidates=[x for x in ev if x['line']<neg['line'] and x['op'] in ('mul.f32','mul.rn.f32') and sorted(x['args'])==sorted(pre['args'][:2])]
    need(len(candidates)==1,'FMA multiplicands differ from maximum-input multiply')
    rounded=candidates[0]
   else:
    need(pre['op'] in ('sub.rn.f32','sub.f32') and len(pre['args'])==2,'expected separate subtract')
    maxreg=pre['args'][1];rounded=current(pre['args'][0],'missing row input')
    if scaled:need(rounded['op'] in ('mul.rn.f32','mul.f32') and not rounded['predicated'],'scaled row not multiply')
   maximum=current(maxreg,'missing max result');need(maximum['op']=='max.f32','prefix maximum not final max tree node')
   paths.append(span(ev,maximum,neg if expect=='SCALED_FMA' else pre,maxreg,True))
   # Structural register tree only. shfl lanes/sync semantics remain unproved.
   def producer(reg,line):
    before=[x for x in ev if x['line']<line and reg in x['dests']];need(before,'missing reduction input');return before[-1]
   leaves=[];nodes=[]
   def visit(r):
    if r['line']==rounded['line']:
     leaves.append(r['line']);return
    need(not r['predicated'],'predicated max tree node')
    need(r['op']=='max.f32' or r['op']=='shfl.sync.bfly.b32','unknown maximum-register-tree node')
    nodes.append(r)
    for reg in r['args'][:2 if r['op']=='max.f32' else 1]:
     q=producer(reg,r['line']);paths.append(span(ev,q,r,reg,True));visit(q)
   visit(maximum);need(leaves,'empty maximum graph')
   exps.append({'graph':expect,'prefix':pre,'rounded_row':rounded,'maximum':maximum,'max_register_tree':nodes,'log2e_multiply':mul,'ex2':row,'checked_straight_spans':paths,'unproved':'Input load/padding and shfl lane/synchronization semantics are not established by this textual register tree.'})
  for name in row['dests']:definitions[name]=row
 need(len(exps)==1,'expected one actual scalar exponent site for B32')
 # Complete source SSA: masked load, optional scale, max, subtract, exp, sum, log, add, store.
 n=r'%[\w.]+';sub=re.search(r'('+n+r') = arith.subf ('+n+r'), ('+n+r') : tensor<32xf32>',ttir);need(sub,'source subtract missing')
 argument,xs,maxsplat=sub.groups();broadcast=re.search(re.escape(maxsplat)+r' = tt.splat ('+n+r') : f32 ->',ttir);need(broadcast,'maximum broadcast missing');maximum=broadcast[1]
 mr=re.search(re.escape(maximum)+r' = "tt.reduce"\('+re.escape(xs)+r'\).*?\n    \}\)',ttir,re.S);need(mr and 'arith.maxnumf' in mr[0],'source maximum no longer maxnum of actual row')
 load=re.search(r'('+n+r') = tt.load ('+n+r'), ('+n+r'), ('+n+r') : tensor<32x!tt.ptr<f32>>',ttir);need(load,'masked fp32 load missing')
 need(re.search(re.escape(load[4])+r' = arith.constant dense<0xFF800000>',ttir),'padding not negative infinity')
 if scaled:
  sc=re.search(re.escape(xs)+r' = arith.mulf '+re.escape(load[1])+r', ('+n+r') :',ttir);need(sc,'actual scaled source multiply missing')
  need(re.search(re.escape(sc[1])+r' = tt.splat %scale : f32 ->',ttir),'scale is not runtime fp32 argument')
 else:need(xs==load[1],'unscaled source no longer direct load')
 exp=re.search(r'('+n+r') = math.exp '+re.escape(argument)+r' :',ttir);need(exp,'exp is not consuming actual sub')
 total=re.search(r'('+n+r') = "tt.reduce"\('+re.escape(exp[1])+r'\).*?\n    \}\)',ttir,re.S);need(total and 'arith.addf' in total[0],'sum does not consume exp')
 log=re.search(r'('+n+r') = math.log '+re.escape(total[1])+r' : f32',ttir);need(log,'log does not consume sum')
 add=re.search(r'('+n+r') = arith.addf '+re.escape(log[1])+r', '+re.escape(maximum)+r' : f32',ttir);need(add,'log/max final addition differs')
 need(re.search(r'tt.store '+n+r', '+re.escape(add[1])+r' : !tt.ptr<f32>',ttir),'final scalar payload not stored')
 # The full actual libdevice-expanded log code is archived. Do not infer a
 # branch-correct semantic slice merely from the following recorded sites.
 logsites=[e for e in ev if e['op'] in ('fma.rn.ftz.f32','selp.f32','setp.eq.f32') or e['control']]
 need(any(e['op']=='fma.rn.ftz.f32' for e in logsites),'expected actual log polynomial FTZ FMA sites missing')
 need(not any(e['op'].startswith('lg2') for e in ev),'unexpected lg2 path')
 return {'source_ssa':{'load':load[1],'mask':load[3],'scaled':xs,'maximum':maximum,'subtract':argument,'exp':exp[1],'sum':total[1],'log':log[1],'store_payload':add[1]},'ptx_exponent_chains':exps,'log_polynomial_and_control_sites':logsites,'log_semantics':'NOT_PROVED; recorded actual FTZ-FMA and guarded logf polynomial path, not a verified log interpreter.','limitations':'Finite straight-register-span checks only; no general PTX/CFG, shuffle, memory or native instruction semantics theorem. Final output numeric equality is not checked.'}
def main():
 receipt=json.loads((N/'receipt.json').read_text());rows=[]
 for r in receipt['records']:
  for rel,h in r['artifact_sha256'].items():need(sha(N/rel)==h,'artifact drift '+rel)
  folder=N/r['id'];result=inspect((folder/'kernel.ptx').read_text(),(folder/'kernel.ttir').read_text(),r['constexprs']['HAS_SCALE'],r['options']['enable_fp_fusion'])
  rows.append({'id':r['id'],'ptx_sha256':sha(folder/'kernel.ptx'),'ttir_sha256':sha(folder/'kernel.ttir'),**result})
 (P/'native-dataflow.json').write_text(json.dumps({'schema':1,'profiles':rows,'profile_count':len(rows),'native_execution':False,'native_receipt_sha256':sha(N/'receipt.json')},indent=2)+'\n')
 print('PASS',len(rows),'profiles')
if __name__=='__main__':main()
