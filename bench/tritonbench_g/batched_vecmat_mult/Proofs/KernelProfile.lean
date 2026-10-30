import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphModel

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelProfile
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult

/-- Source structure available to a native reduction profile. Input tags retain
region/offset identity; neither word nor input leaves expose an FP32 payload. -/
inductive Shape where
  | word
  | input (region : RegionName) (offset : Nat)
  | zero
  | mul (left right : Shape)
  | add (left right : Shape)
  | fma (left right carry : Shape)

def erasePayload : SourceGraph.Expr → Shape
  | .word _ => .word
  | .input r o _ => .input r o
  | .zero => .zero
  | .mul x y => .mul (erasePayload x) (erasePayload y)
  | .add x y => .add (erasePayload x) (erasePayload y)
  | .fma x y z => .fma (erasePayload x) (erasePayload y) (erasePayload z)

/-- Profiles choose valid trees from source structure only. Fixing a profile
and changing runtime FP32 values cannot change the selected reduction plan. -/
abbrev Selector := ∀ n, (Fin n → Shape) → FP.Equational.ReductionPlan n

/-- The consumed adapter to the retained shared proof interface. Public source
execution and the independent native reference both receive this same adapter. -/
def toPlans (profile : Selector) : SourceGraph.PlanSelector :=
  fun n xs => profile n (fun i => erasePayload (xs i))

theorem toPlans_eq (profile : Selector) (xs ys : Fin n → SourceGraph.Expr)
    (same : ∀ i,erasePayload (xs i)=erasePayload (ys i)) :
    toPlans profile n xs=toPlans profile n ys := by
  exact congrArg (profile n) (funext same)

/-- Arbitrary replacement of every floating payload, retaining source tags and
arithmetic structure. This operation is solely for the noninterference proof. -/
def mapPayload (f : ValueProofs.Word → ValueProofs.Word) : SourceGraph.Expr → SourceGraph.Expr
  | .word w => .word (f w)
  | .input r o w => .input r o (f w)
  | .zero => .zero
  | .mul x y => .mul (mapPayload f x) (mapPayload f y)
  | .add x y => .add (mapPayload f x) (mapPayload f y)
  | .fma x y z => .fma (mapPayload f x) (mapPayload f y) (mapPayload f z)

theorem erase_mapPayload (f : ValueProofs.Word → ValueProofs.Word) (x : SourceGraph.Expr) :
    erasePayload (mapPayload f x)=erasePayload x := by
  induction x <;> simp_all [mapPayload,erasePayload]

theorem toPlans_mapPayload (profile : Selector) (f : ValueProofs.Word → ValueProofs.Word)
    (xs : Fin n → SourceGraph.Expr) :
    toPlans profile n (fun i => mapPayload f (xs i))=toPlans profile n xs :=
  toPlans_eq profile _ xs (fun i => erase_mapPayload f (xs i))

#axiomsClean toPlans_eq
#axiomsClean erase_mapPayload
#axiomsClean toPlans_mapPayload
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelProfile
