import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedPointers
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Loaded (c : Config) (s : State α) : Prop where
  value : ∃ v : Values α .fp32 [64,c.BV],s.regs .fp32 [64,c.BV] "b_v"=some v
  dout : ∃ v : Values α .fp32 [64,c.BV],s.regs .fp32 [64,c.BV] "b_do"=some v
  history : ∃ v : Values α .real [c.BV,c.BK],s.regs .real [c.BV,c.BK] "b_h"=some v
  dhist : ∃ v : Values α .real [c.BV,c.BK],s.regs .real [c.BV,c.BK] "b_dh"=some v

def values (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ptrs : Pointers c s) :
    {t : State α // run (engine (withDot M dot)) (((parts c).body.drop 4).take 4) s=some t} := by
  have hvalue := Classical.choose_spec ptrs.value
  have hdout := Classical.choose_spec ptrs.dout
  have hhistory := Classical.choose_spec ptrs.history
  have hdhist := Classical.choose_spec ptrs.dhist
  generalize hvvalue : Classical.choose ptrs.value = vvalue at hvalue
  generalize hvdout : Classical.choose ptrs.dout = vdout at hdout
  generalize hvhistory : Classical.choose ptrs.history = vhistory at hhistory
  generalize hvdhist : Classical.choose ptrs.dhist = vdhist at hdhist
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp only [run,step.eq_2,evalExpr,evalComputeOp]
  simp [evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
    TileShape.axisDim,TileShape.product,Region.cast,State.setReg,hvalue,hdout,hhistory,hdhist]
  rfl

theorem values_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ptrs : Pointers c s) (ready : Ready c s) : Ready c (values M dot c s ptrs).val := by
  constructor
  · simpa [values] using ready.key
  · simpa [values] using ready.time
  · simpa [values] using ready.head
  · simpa [values] using ready.heads
  · simpa [values] using ready.rows
  · simpa [values] using ready.gate
  · simpa [values] using ready.last
  · simpa [values] using ready.dq
  · simpa [values] using ready.dk
  · simpa [values] using ready.ds
  · simpa [values] using ready.dgLast
  · simpa [values] using ready.dg

theorem values_loaded (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ptrs : Pointers c s) : Loaded c (values M dot c s ptrs).val := by
  constructor <;> simp [values]

theorem values_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ptrs : Pointers c s) : (values M dot c s ptrs).val.mem=s.mem := by simp [values]

def loads (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) :
    {t : State α // run (engine (withDot M dot)) ((parts c).body.take 8)
      (s.setReg "i_v" .nat [] (fun _=>block))=some t} := by
  let first := pointers M dot c s ready block
  let last := values M dot c first.val (pointers_available M dot c s ready block)
  refine ⟨last.val,?_⟩
  have split : (parts c).body.take 8=(parts c).body.take 4++((parts c).body.drop 4).take 4 := by rfl
  simpa only [split,run_append,first.property,Option.bind_some] using last.property

theorem loads_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : Ready c (loads M dot c s ready block).val :=
  values_ready M dot c _ (pointers_available M dot c s ready block) (pointers_ready M dot c s ready block)

theorem loads_loaded (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : Loaded c (loads M dot c s ready block).val :=
  values_loaded M dot c _ (pointers_available M dot c s ready block)

theorem loads_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : (loads M dot c s ready block).val.mem=s.mem := by
  change (values M dot c _ (pointers_available M dot c s ready block)).val.mem=s.mem
  rw [values_memory,pointers_memory]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
