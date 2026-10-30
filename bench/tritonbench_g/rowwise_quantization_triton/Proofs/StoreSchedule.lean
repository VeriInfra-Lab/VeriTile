import bench.tritonbench_g.rowwise_quantization_triton.Candidate
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.IOImplements
import VeriTile.Triton.Float.IntegerProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {α : Type} [Inhabited α]

private theorem write_commute (s : State α) (q m : RegionName) (hq : q ≠ m)
    (a b : Nat) (v w : Cell α) :
    (s.write q a v).write m b w = (s.write m b w).write q a v := by
  simp only [State.write]
  congr 1
  funext r o
  by_cases rq : r = q
  · have rm : r ≠ m := by simpa [rq] using hq
    simp [rq,rm,hq]
  · simp [rq]

private theorem fold_write {ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (q m : RegionName) (hq : q ≠ m)
    (off : ι → Nat) (v : ι → Cell α) (b : Nat) (w : Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write q (off i) (v i) else t) s).write m b w =
    xs.foldl (fun t i => if active i then t.write q (off i) (v i) else t) (s.write m b w) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    by_cases hi : active i
    · simp only [if_pos hi,write_commute s q m hq]
    · simp only [if_neg hi]

private theorem fold_regs {ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs=s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons,ih];split <;> rfl

private theorem fold_pids {ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids=s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons,ih];split <;> rfl

/-- Exact full-state equality of the two actual sources. The source-used
llrint/INT8 and FP16 conversion semantics are retained without numerical laws. -/
theorem execution_eq (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (columns width : Nat) (hw : 0 < width) (s : State α) :
    QuantizeIntegerStage.exec (KernelExecution.backend M equal) (some round)
      ⟨Candidate.source "x" "q" "m" columns width,fun _ => none,QuantizeIntegerStage.outputRegister⟩ s =
    QuantizeIntegerStage.exec (KernelExecution.backend M equal) (some round)
      ⟨source "x" "q" "m" columns width,fun _ => none,QuantizeIntegerStage.outputRegister⟩ s := by
  simp (maxSteps := 1500000) [Candidate.source,source,ComputeKernel.surfaceBody,
    QuantizeIntegerStage.exec,QuantizeIntegerStage.run,QuantizeIntegerStage.step,
    QuantizeIntegerStage.evalExpr,QuantizeIntegerStage.store,QuantizeIntegerStage.storedInteger,
    QuantizeIntegerStage.outputRegister,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    FP.Structural.store,numeric,numericLt,numericEq,natLt,bop,KernelExecution.backend,
    State.setReg,ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    TileShape.axisDim,hw,fold_regs,fold_pids]
  rw [fold_write _ _ "q" "m" (by decide)]
  rfl

#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
