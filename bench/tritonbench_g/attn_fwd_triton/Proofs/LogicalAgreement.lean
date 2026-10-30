import bench.tritonbench_g.attn_fwd_triton.Proofs.ViewBounds

/-! Ordinary coordinate-preserving tensor-library contracts suffice for
resolve_neg and packed logical flattening. Only valid input coordinates occur. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.LogicalAgreement
open VeriTile.Triton
noncomputable section

structure Agree (p : Parameters) (xs ys : RealProofs.Inputs) : Prop where
  query : ∀ z < p.batches, ∀ h < p.heads, ∀ n < p.context, ∀ d < p.headDim,
    xs.query z h n d = ys.query z h n d
  key : ∀ z < p.batches, ∀ h < p.heads, ∀ n < p.context, ∀ d < p.headDim,
    xs.key z h n d = ys.key z h n d
  value : ∀ z < p.batches, ∀ h < p.heads, ∀ n < p.context, ∀ d < p.headDim,
    xs.value z h n d = ys.value z h n d
  queryScale : ∀ i < p.batches*p.heads*((p.context+127)/128), xs.queryScale i=ys.queryScale i
  keyScale : ∀ i < p.batches*p.heads*((p.context+63)/64), xs.keyScale i=ys.keyScale i

theorem score (p : Parameters) (xs ys : RealProofs.Inputs) (h : Agree p xs ys)
    (z head m n : Nat) (hz : z<p.batches) (hh : head<p.heads) (hm : m<p.context) (hn : n<p.context) :
    RealProofs.score p.heads p.context p.headDim z head m n xs =
      RealProofs.score p.heads p.context p.headDim z head m n ys := by
  have hq := h.queryScale _ (ViewBounds.packed p.batches p.heads z head _ (m/128) hz hh (by omega))
  have hk := h.keyScale _ (ViewBounds.packed p.batches p.heads z head _ (n/64) hz hh (by omega))
  simp only [RealProofs.score,hq,hk]
  congr 3
  apply List.map_congr_left
  intro d hd
  have hd' := List.mem_range.mp hd
  rw [h.query z hz head hh m hm d hd',h.key z hz head hh n hn d hd']

theorem expected (p : Parameters) (xs ys : RealProofs.Inputs) (h : Agree p xs ys)
    (z head m d : Nat) (hz : z<p.batches) (hh : head<p.heads) (hm : m<p.context) (hd : d<p.headDim) :
    RealProofs.expected p.heads p.context p.headDim z head m d xs =
      RealProofs.expected p.heads p.context p.headDim z head m d ys := by
  have hp : RealProofs.pairs p.heads p.context p.headDim z head m d xs =
      RealProofs.pairs p.heads p.context p.headDim z head m d ys := by
    apply List.map_congr_left
    intro n hn
    have hn' : n<p.context := by have := List.mem_range.mp hn; omega
    rw [score p xs ys h z head m n hz hh hm hn',h.value z hz head hh n hn' d hd]
  simp only [RealProofs.expected,hp]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.LogicalAgreement
