from itertools import product
from pathlib import Path
root=Path('/tmp/veritile-standardization-20261015/kcache-view-wip')
order='abhd'
def mul(*xs):
    return tuple(sorted(sum((tuple(x) for x in xs),()), key=order.index))
def expr(x,base=None):
    fs=list(x)+([base] if base else [])
    if not fs:return '1'
    s=fs[-1]
    for f in fs[-2::-1]:s=f+' * ('+s+')'
    return s

def lst(xs):return '['+','.join(expr(x) for x in xs)+']'

def consume(ns,target):
    ns=list(ns);p=();outs=[];conds=[]
    while ns:
        n=ns[0]
        sp,st=set(p),set(target)
        if p==target:cmp=0
        elif sp<st:cmp=-1
        elif st<sp:cmp=1
        else:raise ValueError((p,target,ns))
        if cmp<0 or not n:
            if cmp<0:conds.append((p,target))
            outs.append(p);p=mul(p,n);ns.pop(0)
        else:
            if cmp>0:raise ValueError('unexpected greater stop')
            break
    return tuple(ns),p,tuple(outs),tuple(dict.fromkeys(conds))

def gtterm(m):
    if len(m)==1:return 'h'+m[0]
    return '(one_lt_mul_of_le_of_lt (le_of_lt h'+m[0]+') '+gtterm(m[1:])+')'
def ltproof(p,t):
    if not p:return 'by exact '+gtterm(t)
    extra=tuple(x for x in t if x not in p)
    return ('by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using '
            'Nat.mul_lt_mul_of_pos_right '+gtterm(extra)+' (show 0 < '+expr(p)+' by positivity)')

proofs=[]
for bits in product([False,True],repeat=4):
    dims={x:(x,) if yes else () for x,yes in zip(order,bits)}
    a,b,h,d=[dims[x] for x in order]
    old=[d,h,b,a];new=[d,h,mul(a,b)]
    tables={};bad=set()
    def go(i,new,chunk):
        chunk=mul(chunk,old[i])
        choices=[True] if i==3 else ([False] if not old[i+1] else [False,True])
        for boundary in choices:
            if not boundary:go(i+1,new,chunk);continue
            key=(tuple(new),chunk)
            result=consume(new,chunk)
            tables[key]=result
            rest,p,_,_=result
            if p!=chunk:bad.add((p,chunk));continue
            if i<3:go(i+1,rest,())
    go(0,new,())
    name='coefficients_'+''.join('L' if x else '1' for x in bits)
    decl='private theorem '+name+' '
    params=[x for x in order if dims[x]]+['sa','sb','sh','sd','st','oh','od']
    decl+='('+' '.join(params)+' : Nat)\n'
    for x in order:
        if dims[x]:decl+=f'    (h{x} : 1 < {x})\n'
    shape=lst([a,b,h,d]);shape2=lst([mul(a,b),h,d])
    decl+=f'    (view : inferViewStrides {shape} [sa,sb,sh,sd] {shape2} = some [st,oh,od]) :\n'
    decl+=f'    ({expr(a)} = 1 ∨ sa = {expr(b)}*st) ∧ ({expr(b)} = 1 ∨ sb = st) ∧\n'
    decl+=f'    ({expr(h)} = 1 ∨ sh = oh) ∧ ({expr(d)} = 1 ∨ sd = od) := by\n'
    if not any(bits):
        decl=decl.replace('(view :', '(_view :')
        decl+='  simp\n'
        proofs.append(decl);continue
    full=mul(a,b,h,d)
    decl+=f'  have hp : {expr(full)} ≠ 0 := by positivity\n'
    for x in order:
        if dims[x]:decl+=f'  have n{x} : {x} ≠ 1 := by omega\n'
    if a and b:decl+=f'  have nab : {expr(mul(a,b))} ≠ 1 := ne_of_gt '+gtterm(mul(a,b))+'\n'
    inequalities=list(dict.fromkeys([c for result in tables.values() for c in result[3]]+[(t,p) for p,t in sorted(bad)]))
    qnames={key:'q'+str(i) for i,key in enumerate(inequalities)}
    for (p,t),q in qnames.items():
        decl+=f'  have {q} : {expr(p)} < {expr(t)} := '+ltproof(p,t)+'\n'
    table_names=[]
    for j,((ns,t),(rest,p,outs,conds)) in enumerate(tables.items()):
        table_names.append('c'+str(j))
        result=f'({lst(rest)}, {expr(p)}, ['+','.join(expr(x,'base') for x in outs)+'])'
        decl+=f'  have c{j} (base : Nat) : consume {lst(ns)} ({expr(t)}) base 1 = {result} := by\n'
        simp=['consume']+[qnames[(p,t)] for p,t in conds]+['n'+x for x in order if dims[x]]+(['nab'] if a and b else [])+['Nat.mul_assoc','Nat.mul_left_comm','Nat.mul_comm']
        decl+='    simp ['+', '.join(simp)+']\n'
    for j,(p,t) in enumerate(sorted(bad)):
        decl+=f'  have e{j} : {expr(p)} ≠ {expr(t)} := ne_of_gt '+qnames[(t,p)]+'\n'
    allrules=table_names+['e'+str(j) for j in range(len(bad))]
    decl+='  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at '+' '.join(allrules)+'\n'
    decl+='  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,\n'
    decl+='    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,\n'
    decl+='    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,\n'
    decl+='    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view\n'
    cases=[]
    if h:cases.append('hs : sh = '+expr(d,'sd'))
    if b:cases.append('hbs : sb = '+(expr(h,'sh') if h else expr(d,'sd')))
    if a:cases.append('has : sa = '+(expr(b,'sb') if b else expr(h,'sh') if h else expr(d,'sd')))
    decl+='  '+' <;> '.join('by_cases '+c for c in cases)+'\n'
    rules=['inferChunks']+allrules+['n'+x for x in order if dims[x]]+[c.split(':')[0].strip() for c in cases]
    decl+='  all_goals simp ['+', '.join(rules)+',\n    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view\n'
    decl+='  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]\n'
    decl+='  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]\n'
    proofs.append(decl)

header='''import bench.tritonbench_g.kcache_copy_triton.HostCorrect

/-! Exhaustive symbolic singleton cases for rank4→rank3 view coefficients.
Each dimension is either exactly one or an arbitrary Nat greater than one.
The local `consume` evaluations are kernel-checked symbolic equations; they
are not finite parameter probes. The companion public address theorem admits
zero dimensions through its empty Fin domain. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

'''
footer='''/-- Every positive rank4→rank3 view preserves the coefficient of each
non-singleton coordinate. The cases above cover the complete singleton/control
flow partition; all non-singleton sizes and all strides remain arbitrary. -/
theorem coefficients_positive (a b h d sa sb sh sd st oh od : Nat)
    (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) (hd : 0 < d)
    (view : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a*b,h,d] = some [st,oh,od]) :
    (a = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  rcases (show a = 1 ∨ 1 < a by omega) with rfl | ha <;>
    rcases (show b = 1 ∨ 1 < b by omega) with rfl | hb <;>
    rcases (show h = 1 ∨ 1 < h by omega) with rfl | hh <;>
    rcases (show d = 1 ∨ 1 < d by omega) with rfl | hd
'''
for bits in product([False,True],repeat=4):
    name='coefficients_'+''.join('L' if x else '1' for x in bits)
    names=[x for x,yes in zip(order,bits) if yes]+['sa','sb','sh','sd','st','oh','od']+['h'+x for x,yes in zip(order,bits) if yes]
    footer+='  · simpa only [Nat.one_mul,Nat.mul_one] using '+name+' '+' '.join(names)+'\n      (by simpa only [Nat.one_mul,Nat.mul_one] using view)\n'
footer+='''
#axiomsClean coefficients_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
'''
(root/'bench/tritonbench_g/kcache_copy_triton/Proofs/HostViewCases.lean').write_text(header+'\n'.join(proofs)+'\n'+footer)
print('Generated',len(proofs),'cases,',sum(len(p.splitlines()) for p in proofs),'lines')
