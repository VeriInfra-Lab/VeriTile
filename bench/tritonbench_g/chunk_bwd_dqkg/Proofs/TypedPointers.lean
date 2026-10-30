import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedEval
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Pointers (c : Config) (s : State α) : Prop where
  value : ∃ v : Values α .blockPtr [64,c.BV],s.regs .blockPtr [64,c.BV] "p_v"=some v
  dout : ∃ v : Values α .blockPtr [64,c.BV],s.regs .blockPtr [64,c.BV] "p_do"=some v
  history : ∃ v : Values α .blockPtr [c.BV,c.BK],s.regs .blockPtr [c.BV,c.BK] "p_h"=some v
  dhist : ∃ v : Values α .blockPtr [c.BV,c.BK],s.regs .blockPtr [c.BV,c.BK] "p_dh"=some v

def pointers (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) :
    {t : State α // run (engine (withDot M dot)) ((parts c).body.take 4)
      (s.setReg "i_v" .nat [] (fun _=>block))=some t} := by
  have hkey := Classical.choose_spec ready.key
  have htime := Classical.choose_spec ready.time
  have hhead := Classical.choose_spec ready.head
  generalize hvkey : Classical.choose ready.key = key at hkey
  generalize hvtime : Classical.choose ready.time = time at htime
  generalize hvhead : Classical.choose ready.head = head at hhead
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp only [run,step.eq_2,evalExpr,evalComputeOp]
  simp [eval_block,eval_mul,eval_add,eval_nat,eval_ref,evalOffsets,
    Option.bind,numeric,bop,State.setReg,hkey,htime,hhead]
  rfl

theorem pointers_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : Ready c (pointers M dot c s ready block).val := by
  constructor
  · simpa [pointers] using ready.key
  · simpa [pointers] using ready.time
  · simpa [pointers] using ready.head
  · simpa [pointers] using ready.heads
  · simpa [pointers] using ready.rows
  · simpa [pointers] using ready.gate
  · simpa [pointers] using ready.last
  · simpa [pointers] using ready.dq
  · simpa [pointers] using ready.dk
  · simpa [pointers] using ready.ds
  · simpa [pointers] using ready.dgLast
  · simpa [pointers] using ready.dg

theorem pointers_available (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : Pointers c (pointers M dot c s ready block).val := by
  constructor <;> simp [pointers]

theorem pointers_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : (pointers M dot c s ready block).val.mem=s.mem := by simp [pointers]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
