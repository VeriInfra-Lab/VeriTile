import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.StoredReal

/-! Independent instruction graph for the selected FP32 kernels. Unlike the
historical graph, forward retains the initial zero additions in both source
accumulators, and count divisions are always present (also at N=1).
Primitive operations and compiler contraction/reduction choices are ordinary
model parameters; no numerical experiment is an assumption of these theorems.
This graph is not asserted to be the semantics of the typed source until that
separate refinement is proved. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.KernelGraph
open VeriTile.Bench.Repairs.FastLayernorm (MulAdd)
noncomputable section

structure ForwardPlan (N B : Nat) where
  base : Plan N B
  varianceTree : Nat → SumTree B
  varianceComplete : ∀ lane,(varianceTree lane).leaves.Perm (List.finRange B)
  varianceKind : MulAdd

/-- One source streaming iteration; wrapper dispatch has 0<N≤B. Every leaf
performs `0 + load`, including masked zero leaves. -/
def mean (ops : Ops α) (p : ForwardPlan N B) (x : Nat → α) (lane : Nat) : α :=
  ops.div ((p.base.meanTree lane).eval ops (fun j => ops.add ops.zero (masked ops N x j))) (ops.count N)

def centered (ops : Ops α) (p : ForwardPlan N B) (x : Nat → α) (j : Fin B) : α :=
  if j.val<N then ops.sub (x j.val) (mean ops p x j.val) else ops.zero

/-- `varianceKind` preserves the selected multiply/add or FMA at the source
`var_acc += x*x`; the additive zero remains an actual operand. -/
def variance (ops : Ops α) (p : ForwardPlan N B) (x : Nat → α) (lane : Nat) : α :=
  ops.div ((p.varianceTree lane).eval ops
    (fun j => mulAdd ops p.varianceKind (centered ops p x j) (centered ops p x j) ops.zero)) (ops.count N)

def stats (ops : Ops α) (p : ForwardPlan N B) (x : Nat → α) (eps : α) (lane : Nat) : Reuse.Statistics α :=
  ⟨mean ops p x lane,
   ops.div ops.one (ops.sqrt (ordered ops.add p.base.epsilonSwap (variance ops p x lane) eps))⟩

def output (ops : Ops α) (p : ForwardPlan N B) (x w b : Nat → α) (eps : α) (col : Nat) : α :=
  mulAdd ops p.base.outputKind (normed ops p.base (stats ops p x eps col) (x col)) (w col) (b col)

/-- Native backward never deletes division by one. Existing row, first-row,
partial-storage and streaming-final graphs are reused unchanged. -/
structure BackwardPlan (N B : Nat) where
  base : Plan N B
  divisionRetained : base.eraseUnitDivision=Bool.false
  final : FinalPlan

def backward := @backwardValue

theorem mean_real (p : ForwardPlan N B) (x : Nat → ℝ) (lane : Nat) :
    mean RealProof.ops p x lane=RealProof.Spec.mean N x := by
  have he : (fun j : Fin B => RealProof.ops.add RealProof.ops.zero (masked RealProof.ops N x j)) =
      masked RealProof.ops N x := by funext j; simp [RealProof.ops]
  unfold mean
  rw [he,RealProof.sum_padded _ (p.base.meanComplete lane) p.base.covers]
  rfl

theorem centered_real (p : ForwardPlan N B) (x : Nat → ℝ) :
    centered RealProof.ops p x=masked RealProof.ops N (fun j => x j-RealProof.Spec.mean N x) := by
  funext j
  unfold centered
  rw [mean_real]
  rfl

theorem variance_real (p : ForwardPlan N B) (x : Nat → ℝ) (lane : Nat) :
    variance RealProof.ops p x lane=RealProof.Spec.variance N x := by
  have he : (fun j : Fin B => mulAdd RealProof.ops p.varianceKind
      (centered RealProof.ops p x j) (centered RealProof.ops p x j) RealProof.ops.zero)=
      masked RealProof.ops N (fun j => (x j-RealProof.Spec.mean N x)^2) := by
    funext j
    rw [RealProof.mulAdd_real,centered_real]
    simp only [masked,RealProof.ops]
    split <;> simp [pow_two]
  unfold variance
  rw [he,RealProof.sum_padded _ (p.varianceComplete lane) p.base.covers]
  rfl

theorem stats_real (p : ForwardPlan N B) (x : Nat → ℝ) (eps : ℝ) (lane : Nat) :
    stats RealProof.ops p x eps lane=
      ⟨RealProof.Spec.mean N x,RealProof.Spec.inverse N x eps⟩ := by
  unfold stats
  rw [mean_real,variance_real,RealProof.ordered_add]
  rfl

theorem output_real (p : ForwardPlan N B) (x w b : Nat → ℝ) (eps : ℝ) (col : Nat) :
    output RealProof.ops p x w b eps col=RealProof.Spec.output N x w b eps col := by
  unfold output
  rw [stats_real,RealProof.mulAdd_real,RealProof.normed_real]
  rfl

#axiomsClean mean_real
#axiomsClean variance_real
#axiomsClean stats_real
#axiomsClean output_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.KernelGraph
