import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedLoads
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def accum (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (loaded : Loaded c s) :
    {t : State α // run (engine (withDot M dot)) ((parts c).body.drop 8) s=some t} := by
  have hdq := Classical.choose_spec ready.dq
  have hdk := Classical.choose_spec ready.dk
  have hds := Classical.choose_spec ready.ds
  have hdgLast := Classical.choose_spec ready.dgLast
  have hvalue := Classical.choose_spec loaded.value
  have hdout := Classical.choose_spec loaded.dout
  have hhistory := Classical.choose_spec loaded.history
  have hdhist := Classical.choose_spec loaded.dhist
  generalize hvdq : Classical.choose ready.dq = vdq at hdq
  generalize hvdk : Classical.choose ready.dk = vdk at hdk
  generalize hvds : Classical.choose ready.ds = vds at hds
  generalize hvdgLast : Classical.choose ready.dgLast = vdgLast at hdgLast
  generalize hvvalue : Classical.choose loaded.value = vvalue at hvalue
  generalize hvdout : Classical.choose loaded.dout = vdout at hdout
  generalize hvhistory : Classical.choose loaded.history = vhistory at hhistory
  generalize hvdhist : Classical.choose loaded.dhist = vdhist at hdhist
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp only [run,step.eq_2,evalExpr,evalComputeOp]
  simp [evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
    TileShape.axisDim,TileShape.product,Region.cast,State.setReg,hdq,hdk,hds,hdgLast,hvalue,hdout,hhistory,hdhist]
  rfl

theorem accum_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (loaded : Loaded c s) : Ready c (accum M dot c s ready loaded).val := by
  constructor
  · simpa [accum] using ready.key
  · simpa [accum] using ready.time
  · simpa [accum] using ready.head
  · simpa [accum] using ready.heads
  · simpa [accum] using ready.rows
  · simpa [accum] using ready.gate
  · simpa [accum] using ready.last
  · simp [accum]
  · simp [accum]
  · simp [accum]
  · simp [accum]
  · simpa [accum] using ready.dg

theorem accum_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (loaded : Loaded c s) : (accum M dot c s ready loaded).val.mem=s.mem := by simp [accum]

def body (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : State α :=
  (accum M dot c (loads M dot c s ready block).val
    (loads_ready M dot c s ready block) (loads_loaded M dot c s ready block)).val

theorem body_runs (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) :
    run (engine (withDot M dot)) (parts c).body (s.setReg "i_v" .nat [] (fun _=>block))=
      some (body M dot c s ready block) := by
  rw [← List.take_append_drop 8 (parts c).body,run_append]
  simp only [(loads M dot c s ready block).property,Option.bind_some]
  exact (accum M dot c (loads M dot c s ready block).val
    (loads_ready M dot c s ready block) (loads_loaded M dot c s ready block)).property

theorem body_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : Ready c (body M dot c s ready block) :=
  accum_ready M dot c _ (loads_ready M dot c s ready block) (loads_loaded M dot c s ready block)

theorem body_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) (block : Nat) : (body M dot c s ready block).mem=s.mem := by
  unfold body
  rw [accum_memory,loads_memory]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
