import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Prologue

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

structure Invariant (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial : RegisterState α) (k : Nat) (current : RegisterState α) : Prop where
  memory : current.mem=initial.mem
  pids : current.pids=initial.pids
  numPids : current.numPids=initial.numPids
  maskM : p.divM=Bool.false → current.regs .bool [p.TM] "mask_m"=
    some (fun i=>decide (row p initial i.1<p.M))
  maskN : p.divN=Bool.false → current.regs .bool [p.TN] "mask_n"=
    some (fun i=>decide (column p initial i.1<p.N))
  offsk : current.regs .nat [p.TK] "offs_k"=some (fun i=>inner p k i.1)
  aptrs : current.regs .ptr [p.TM,p.TK] "a_ptrs"=some (fun i=>(p.A,aOffset p initial k i))
  bptrs : current.regs .ptr [p.TK,p.TN] "b_ptrs"=some (fun i=>(p.B,bOffset p initial k i))
  optrs : current.regs .ptr [p.TM,p.TN] "o_ptrs"=some (fun i=>(p.O,oOffset p initial i))
  iterations : current.regs .nat [] "num_iters"=some (fun _=>iterations p)
  acc : current.regs .real [p.TM,p.TN] "o"=some (accumulation o p initial k)

theorem preludeInvariant (o : BmmOptimized.Scalar α) (p : Parameters) (s : RegisterState α) :
    Invariant o p s 0 (preludeResult o p s) := by
  cases hm : p.divM <;> cases hn : p.divN <;> by_cases hg : p.GROUP=1
  all_goals constructor <;>
    simp [preludeResult,tailResult,gridResult,hm,hn,hg,accumulation,inner]

theorem loopRegister (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial current : RegisterState α) (k : Nat) (h : Invariant o p initial k current) :
    Invariant o p initial k (current.setReg "_i" .nat [] (fun _=>k)) := by
  constructor <;> simp [h.memory,h.pids,h.numPids,h.offsk,h.aptrs,h.bptrs,h.optrs,h.iterations,h.acc]
  · exact h.maskM
  · exact h.maskN

#axiomsClean preludeInvariant
#axiomsClean loopRegister
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
