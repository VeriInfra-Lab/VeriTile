import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
open VeriTile.Triton Whole
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

namespace HeadlineMath
open Spec
def baseQ (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  (∑ j ∈ Finset.range c.V, vd c a.outputGrad h chunk r j * hist c a.history h chunk e j) *
    Real.exp (gate c a h chunk r) * c.scale
def baseK (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  (∑ j ∈ Finset.range c.V, vd c a.value h chunk r j * hist c a.historyGrad h chunk e j) *
    Real.exp (0-gate c a h chunk r+lastGate c a h chunk)
def score (c : Config) (a : Tensors) (h chunk r u : Nat) : ℝ :=
  if u≤r then
    (∑ j ∈ Finset.range c.V, vd c a.outputGrad h chunk r j * vd c a.value h chunk u j) *
      c.scale * Real.exp (gate c a h chunk r-gate c a h chunk u)
  else 0
def dq (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  baseQ c a h chunk r e + ∑ u : Fin 64, score c a h chunk r u.val * qk c a.key h chunk u.val e
def dk (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  baseK c a h chunk r e + ∑ u : Fin 64, score c a h chunk u.val r * qk c a.query h chunk u.val e
def correction (c : Config) (a : Tensors) (key h chunk : Nat) : ℝ :=
  (∑ j ∈ Finset.range c.V, ∑ e : Fin c.BK,
    hist c a.history h chunk (key*c.BK+e.val) j * hist c a.historyGrad h chunk (key*c.BK+e.val) j) *
    Real.exp (lastGate c a h chunk) +
  ∑ r : Fin 64, ∑ e : Fin c.BK,
    baseK c a h chunk r.val (key*c.BK+e.val) * qk c a.key h chunk r.val (key*c.BK+e.val)
def dg (c : Config) (a : Tensors) (key h chunk r : Nat) : ℝ :=
  (∑ e : Fin c.BK,
    (qk c a.query h chunk r (key*c.BK+e.val) * dq c a h chunk r (key*c.BK+e.val) -
    qk c a.key h chunk r (key*c.BK+e.val) * dk c a h chunk r (key*c.BK+e.val))) +
  if r<min 64 (c.T-chunk*64)-1 then 0 else correction c a key h chunk
end HeadlineMath

/-- Named dQ/dK tensors and the unreduced dG tensor for every key tile. -/
def logicalFormula (c : Config) (xs : Logical.Inputs c) : Logical.Outputs c where
  dq := fun p => HeadlineMath.dq c xs.tensors p.1.val (p.2.1.val/64) (p.2.1.val%64) p.2.2.val
  dk := fun p => HeadlineMath.dk c xs.tensors p.1.val (p.2.1.val/64) (p.2.1.val%64) p.2.2.val
  dg := fun p => HeadlineMath.dg c xs.tensors p.1.val p.2.1.val (p.2.2.val/64) (p.2.2.val%64)

/-- Complete source grid, all three logical outputs, actual source access
safety, successful execution and exact exterior memory frame. -/
specification kernel_real (c : Config) (order : List (CTA c))
    (ho : order.Perm (allCTAs c)) :
    Spec.Real (Logical.io c order ⊨ logicalFormula c) :=
  Logical.logical_implements c order ho

#axiomsClean kernel_real
#specNonCircular logicalFormula avoiding [Whole.runGrid,VeriTile.Triton.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg
