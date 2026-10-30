from inspect_ptx import scan,BASE
from pathlib import Path
import json

class Unrelated(Exception): pass

def norm_expected(diagonal,owner=0,tiny=False):
    def local(u):
        a,b,c,d=[4*u+i for i in range(4)]
        if diagonal:
            if tiny: return ('add',('product',d),('add',('product',c),('add',('product',a),('product',b))))
            return ('add',('add',('add',('product',a),('product',b)),('product',c)),('product',d))
        return ('fma',d,('fma',c,('fma',a,('product',b))))
    def shuffle(stage,u):
        if not stage: return local(u)
        distance=[4,2,1][stage-1]
        return ('add',shuffle(stage-1,u),shuffle(stage-1,u^distance))
    return shuffle(3,owner)

def tree_leaves(t):
    if t[0]=='product': return [t[1]]
    if t[0]=='fma': return [t[1]]+tree_leaves(t[2])
    if t[0]=='add': return tree_leaves(t[1])+tree_leaves(t[2])
    raise ValueError(t)

def audit_norm(path,tiny=False):
    events=scan(path); result=[]
    for diagonal,source_square,source_reduce in [(False,56,57),(True,82,84)]:
        squares=[e for e in events if e['loc']==source_square and e['op']=='mul.f32' and e['node'][1]==e['node'][2]]
        raw={e['node'][1]:i for i,e in enumerate(squares)}
        assert len(raw)==32,(path,len(raw),source_square)
        def term(n,lane):
            if n[0]=='mul.f32' and n[1]==n[2] and n[1] in raw:
                idx=raw[n[1]];return ('product',4*lane+idx%4),idx//4,False
            if n[0] in ('selp.f32','selp.b32'):
                nonzero=[x for x in n[1:3] if x not in [('input','0'),('input','0f00000000')]]
                if len(nonzero)!=1: raise Unrelated()
                t,row,masked=term(nonzero[0],lane)
                if t[0]!='product' or masked: raise Unrelated()
                return t,row,True
            if n[0]=='fma.rn.f32' and n[1]==n[2] and n[1] in raw:
                t,row,masked=term(n[3],lane);idx=raw[n[1]]
                if row!=idx//4 or masked: raise Unrelated()
                return ('fma',4*lane+idx%4,t),row,masked
            if n[0]=='add.f32':
                a,ar,am=term(n[1],lane);b,br,bm=term(n[2],lane)
                if ar!=br or am!=bm: raise Unrelated()
                return ('add',a,b),ar,am
            if n[0]=='shfl.sync.bfly.b32':
                if n[3:]!=(('input','31'),('input','-1')): raise Unrelated()
                distance=int(n[2][1])
                if distance not in (1,2,4): raise Unrelated()
                return term(n[1],lane^distance)
            raise Unrelated()
        roots=[]
        for e in events:
            if e['loc']!=source_reduce or e['op']!='add.f32': continue
            try: t,row,masked=term(e['node'],0)
            except Unrelated: continue
            if len(tree_leaves(t))!=32: continue
            assert t==norm_expected(diagonal,tiny=tiny),(path,e['line'],'norm tree mismatch')
            assert masked==diagonal
            # Check every physical column owner, not just one representative.
            for owner in range(8):
                other,otherrow,othermasked=term(e['node'],owner)
                assert other==norm_expected(diagonal,owner,tiny)
                assert otherrow==row and othermasked==masked
            roots.append({'row_register':row,'ptx_line':e['line'],'destination':e['dst']})
        assert sorted(x['row_register'] for x in roots)==list(range(8)),(path,source_reduce,roots)
        result.append({'diagonal':diagonal,'roots':roots,'owners_checked':8,'leaf_permutation':tree_leaves(norm_expected(diagonal,tiny=tiny)),
                       'tree':norm_expected(diagonal,tiny=tiny)})
    return result

if __name__=='__main__':
    for path in sorted(BASE.glob('actual-binder-*-forward/kernel.ptx')):
        result=audit_norm(path,tiny='tiny' in path.parent.name)
        print(path.parent.name,[(x['diagonal'],len(x['roots'])) for x in result])
