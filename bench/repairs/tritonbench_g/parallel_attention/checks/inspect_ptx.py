from pathlib import Path
import re,json,collections,sys

BASE=Path('/tmp/parallel-attention-lowering')
DOT_LINES={55,60,81,86,149,154,156,175,181,185,229,235,236,243,258,263,270,272}

def scan(path):
    regs={}; events=[]; loc=0; last_dot=0; serial=0; debug=''
    def read(x): return regs.get(x, ('input',x))
    for line,s in enumerate(path.read_text().splitlines(),1):
        if '.loc' in s:
            debug=s.strip()
            m=re.search(r'parallel_attention_repaired.py:(\d+)',s)
            loc=int(m[1]) if m else -1
        m=re.match(r'\s*(fma\.rn\.f32|mul\.f32|add\.f32|selp\.(?:f32|b32)|mov\.b32|shfl\.sync\.bfly\.b32)\s+([^;]+);',s)
        if not m: continue
        op=m[1]; args=[x.strip() for x in m[2].split(',')]; dst=args[0]; values=[read(x) for x in args[1:]]
        if op=='fma.rn.f32' and (loc in DOT_LINES or (loc==0 and values[2][0]=='dot')):
            old=values[2]
            dotloc=loc if loc in DOT_LINES else old[1]
            if old[0]=='dot' and old[1]==dotloc:
                node=('dot',dotloc,old[2]+1,old[3],old[4],line)
            else:
                serial+=1
                node=('dot',dotloc,1,old,line,line)
        else: node=(op,*values)
        regs[dst]=node
        events.append({'line':line,'loc':loc,'debug':debug,'op':op,'dst':dst,'args':args[1:],'node':node})
    return events

def compact(n):
    if n[0]=='dot': return ['dot',n[1],n[2],compact(n[3]),n[4],n[5]]
    if n[0]=='input': return n[1]
    return [n[0],*[compact(a) for a in n[1:]]]

if __name__=='__main__':
    for name in sys.argv[1:] or ['ordinary','ragged','tiny','sharded','noscale-ragged','noscale-sharded']:
        print('\nPROFILE',name)
        events=scan(BASE/f'actual-binder-{name}-backward/kernel.ptx')
        groups=collections.defaultdict(list)
        for e in events:
            if e['loc'] in [134,156,160,180,184,230,238,241,242,258,268,271] and e['node'][0]!='dot':
                groups[(e['loc'],e['op'])].append(e)
        for key,es in sorted(groups.items()):
            for e in es[:1]: print(key,len(es),e['line'],json.dumps(compact(e['node'])))
