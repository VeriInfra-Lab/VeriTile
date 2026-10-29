import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePair
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePairSafety
open VeriTile.Triton FP.Structural SourcePair
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false


theorem trace_congr_steps {α : Type} [Inhabited α] (A B : Algebra α)
    (bounds : RegionName → Nat) {left right : List ComputeStmt}
    (same : List.Forall₂ (fun a b => ∀ s, step A a s = step B b s) left right)
    (safe : List.Forall₂ (fun a b => ∀ s, Safety.stepSafe A bounds a s ↔
      Safety.stepSafe B bounds b s) left right) (s : State α) :
    Safety.traceSafeList A bounds left s ↔ Safety.traceSafeList B bounds right s := by
  induction same generalizing s with
  | nil => simp only [Safety.traceSafeList_unfold]
  | cons head tail ih =>
    cases safe with
    | cons hsafe tsafe =>
      simp only [Safety.traceSafeList_cons,head,hsafe,ih tsafe]


theorem append_steps {α β : Type} {R : α → β → Prop}
    {a c : List α} {b d : List β} (left : List.Forall₂ R a b)
    (right : List.Forall₂ R c d) : List.Forall₂ R (a++c) (b++d) := by
  induction left with
  | nil => exact right
  | cons h _ ih => exact .cons h ih


end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePairSafety
