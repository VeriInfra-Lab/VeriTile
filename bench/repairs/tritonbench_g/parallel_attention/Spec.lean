import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativePlans
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Software
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Types

namespace VeriTile.Bench.Repairs.ParallelAttention.Spec
noncomputable section

/-- Ordered FP reference for a logical element: finite scalar dot/carry/reduce
syntax selected by the structural compiler checker, with no memory execution. -/
def raw (a : Ops α) (c : CompilerLayout) (s : Shape) (trees : OutputKind → SumPlan)
    (x : TensorInputs α) (kind : OutputKind) (bh t e : Nat) : α :=
  nativeWordReference a c s kind (trees kind) x (dense s.T (kind.width s) bh t e)

def forward (a : Ops α) (c : CompilerLayout) (s : Shape) (trees : OutputKind → SumPlan)
    (x : TensorInputs α) (eps : α) (both normalize : Bool) (bh : Nat) : PublicResult α :=
  let o:=raw a c s trees x .forward bh
  let z:=fun t=>raw a c s trees x .normalizer bh t 0
  if both then .both o z
  else .single (if normalize then fun t e=>a.div (o t e) (a.add (z t) eps) else o)

def Agrees (T V : Nat) : PublicResult α → PublicResult α → Prop
  | .both o z,.both o' z' => (∀ t,t<T → ∀ e,e<V → o t e=o' t e) ∧ (∀ t,t<T → z t=z' t)
  | .single o,.single o' => ∀ t,t<T → ∀ e,e<V → o t e=o' t e
  | _,_ => False

#specNonCircular forward avoiding [compiledRun,combinedRun,combinedScratch,combinedHost,runTagged,runStores]
end
end VeriTile.Bench.Repairs.ParallelAttention.Spec
