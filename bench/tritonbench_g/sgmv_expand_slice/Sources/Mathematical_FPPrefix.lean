import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def aPointer [Inhabited Word] (p : Parameters) (s : State Word) (cur : Nat) :
    Values Word .ptr [p.blockRows,p.blockInner] := fun i => (p.input,aOffset p s cur i)
def bPointer [Inhabited Word] (p : Parameters) (s : State Word) (cur : Nat) :
    Values Word .ptr [p.blockInner,p.blockColumns] := fun i => (p.weight,bOffset p s cur i)

structure Ready [Inhabited Word] (p : Parameters) (s t : State Word) (cur : Nat)
    (acc : Accumulator p) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  pidm : t.regs .nat [] "pid_m"=some (fun _ => s.pids 0)
  pidn : t.regs .nat [] "pid_n"=some (fun _ => s.pids 1)
  batch : t.regs .nat [] "cur_batch"=some (fun _ => s.pids 2)
  length : t.regs .nat [] "M"=some (fun _ => seqLen p s)
  start : t.regs .nat [] "cur_seq_start"=some (fun _ => seqStart p s)
  adapter : t.regs .nat [] "lora_index"=some (fun _ => adapter p s)
  offm : t.regs .nat [p.blockRows] "offset_m"=some (fun i => row p s i.1)
  offn : t.regs .nat [p.blockColumns] "offset_n"=some (fun i => col p s i.1)
  offk : t.regs .nat [p.blockInner] "offset_k"=some (fun i => i.1.val)
  ram : t.regs .nat [p.blockRows] "ram"=some (fun i => row p s i.1 % seqLen p s)
  rbn : t.regs .nat [p.blockColumns] "rbn"=some (fun i => col p s i.1 % p.columns)
  aptr : t.regs .ptr [p.blockRows,p.blockInner] "a_ptr"=some (aPointer p s cur)
  bptr : t.regs .ptr [p.blockInner,p.blockColumns] "b_ptr"=some (bPointer p s cur)
  acc : t.regs .real [p.blockRows,p.blockColumns] "accumulator"=some acc

end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
