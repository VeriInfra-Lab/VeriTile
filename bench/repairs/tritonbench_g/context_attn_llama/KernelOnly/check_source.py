"""Actual-source CPU replay and complete physical backing checks."""
from pathlib import Path
import hashlib
import json
import runpy

TASK=Path(__file__).resolve().parents[1]
ROOT=TASK.parents[3]
OUT=ROOT/".veritile-cache/kernel-evidence/context_attn_llama"
SOURCE=TASK/"context_attn_llama_repaired.py"
PIN="8317f23a4c8c04305b230369c2f59f6c4aaf41a5e80dca5336e88a1e13617e0b"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PIN
base=runpy.run_path(str(TASK/"check_contract.py"))
torch,repair=base["torch"],base["repair"]
Tensor,Device,Launch=base["Tensor"],base["Device"],base["Launch"]

def backing(t):
    return torch.empty(0,dtype=torch.uint8).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,))

records=[]
def check(width,length,prompt,tile,alias=False,exceptional=False):
    rows,heads,kvheads=length+4,2,1
    if alias: heads=1
    cache=rows if alias else 19
    qb=torch.randn(rows+4,heads,width)*.1
    kb=torch.randn(cache+4,kvheads,width)*.1
    vb=torch.randn_like(kb)*.1
    q,k,v=qb[2:-2],kb[2:-2],vb[2:-2]
    ob=torch.full((rows+4,heads,width),71.)
    out=ob[2:-2]
    tokens=length+prompt+3
    ints=[torch.tensor([911,x,913],dtype=torch.int32)[1:2] for x in [0,1,length+prompt,prompt]]
    tb=torch.randint(cache,(3,tokens),dtype=torch.int32)
    table=tb[1:2]
    if alias:
        k=v=q
        allints=q.view(torch.int32).flatten()
        allints[:4]=torch.tensor([0,1,length+prompt,prompt],dtype=torch.int32)
        allints[4:4+tokens]=torch.arange(tokens,dtype=torch.int32)%cache
        ints=[allints[i:i+1] for i in range(4)]
        table=allints[4:4+tokens].reshape(1,tokens)
    if exceptional:
        q[1,0,0]=float("nan")
        k[0,0,1]=float("inf")
        v[0,0,2]=-0.0
    reads=(q,k,v,*ints,table)
    snapshots=[backing(t).clone() for t in reads]
    before=backing(out).clone()
    Device.name="Tesla interpreter facade" if tile==64 else "CPU interpreter facade"
    count=Launch.count
    assert repair.context_attention_fwd(*[Tensor(t) for t in (q,k,v,out,*ints)],length+13,Tensor(table)) is None
    assert Launch.count-count==int(length>0)
    assert Device.current==torch.device("cuda:1")
    assert all(torch.equal(backing(t),s) for t,s in zip(reads,snapshots))
    touched=torch.zeros_like(before,dtype=torch.bool)
    begin=(out.storage_offset()+heads*width)*4
    touched[begin:begin+length*heads*width*4]=True
    assert torch.equal(backing(out)[~touched],before[~touched])
    if not exceptional:
        scale=torch.tensor(1.0/(width**.5)*1.4426950408889634,dtype=torch.float32)
        for row in range(length):
            end=min((row//tile+1)*tile+prompt,length+prompt)
            rounded=(end+tile-1)//tile*tile
            for head in range(heads):
                keys=torch.zeros(rounded,width)
                values=torch.zeros_like(keys)
                keys[:end]=k[table[0,:end].long(),head//(heads//kvheads)]
                values[:end]=v[table[0,:end].long(),head//(heads//kvheads)]
                scores=(keys@q[1+row,head])*scale
                scores[torch.arange(rounded)>row+prompt]=-1.e8
                weights=torch.exp2(scores-scores.max())
                expected=(weights@values)/weights.sum()
                torch.testing.assert_close(out[1+row,head],expected,rtol=7e-4,atol=1e-5)
    records.append({"dimension":width,"length":length,"prompt":prompt,"tile":tile,
        "payload_metadata_page_table_alias":alias,"NaN_Inf_signed_zero_admitted":exceptional,
        "complete_read_backings_and_output_complement":True})

torch.manual_seed(19)
for args in [(16,63,2,64),(32,64,1,64),(64,65,7,64),(128,127,2,128),(256,128,1,128),(16,129,7,128),(16,0,3,64)]:
    check(*args)
check(16,3,2,64,alias=True)
check(16,3,2,128,exceptional=True)
receipt={"source":str(SOURCE.relative_to(ROOT)),"source_sha256":PIN,
    "historical_actual_source_and_guard_controls":"passed","additional_complete_source_cases":records,
    "additional_complete_source_executions":len(records),"total_nonempty_launches_in_this_process":Launch.count,
    "GPU_execution":False,"native_instruction_numerical_conformance":False}
(OUT/"source-controls.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"additional_complete_source_executions":len(records),"full_backing_frames":"passed"}))
