import bench.repairs.tritonbench_g.l2_norm_triton2.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Scalar

namespace VeriTile.Bench.Repairs.L2NormTriton2
open VeriTile.Triton
theorem backward_length (N B : Nat) : (backward N B).surfaceBody.length=14 := by rfl

theorem backward_io (N B : Nat) :
    (backward N B).inputs=["X","DY"] ∧ (backward N B).outputs=["DX"] := ⟨rfl,rfl⟩

end VeriTile.Bench.Repairs.L2NormTriton2

namespace VeriTile.Bench.Repairs.L2NormTriton2.Tile
open VeriTile.Triton FP.Structural
noncomputable section
abbrev loaded := @L2NormTriton1.Tile.loaded
abbrev masked := @L2NormTriton1.Tile.masked
abbrev variance := @L2NormTriton1.Tile.variance

def dot {α : Type} (M : Algebra α) (N B : Nat) (xs dys : Nat → α) : α :=
  (M.reduceSum (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false
    (FP.Structural.bop (M.binary (some .fp32) .real .mul)
      (Broadcast.consSame Broadcast.nil)
      (fun i => masked M N B dys i.1) (fun i => masked M N B xs i.1))) PUnit.unit

/-- Both source reductions, casts, and left-associated scaling remain explicit.
No mathematical or floating-point simplification enters this operational row. -/
def scalar {α : Type} (M : Algebra α) (N B : Nat) (eps : α) (xs dys : Nat → α)
    (i : Nat) : α :=
  let cast := M.cast (some .fp32) .real .real
  let bin := M.binary (some .fp32) .real
  let shifted := bin .add (variance M N B xs) eps
  let one := M.literal (some .fp32) .real 1
  let rstd := bin .div one (M.unary (some .fp32) .sqrt shifted)
  bin .sub (bin .mul (cast (dys i)) rstd)
    (bin .mul (bin .mul (bin .mul (dot M N B xs dys) (bin .div one shifted)) rstd) (cast (xs i)))

theorem dot_congr {α : Type} (M : Algebra α) (N B : Nat) (xs dys ys dzs : Nat → α)
    (hx : ∀ i,i<N → xs i=ys i) (hd : ∀ i,i<N → dys i=dzs i) :
    dot M N B xs dys=dot M N B ys dzs := by
  unfold dot
  congr 1
  funext i
  simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  by_cases hi : i.1.val<N
  · simp [masked,loaded,L2NormTriton1.Tile.masked,L2NormTriton1.Tile.loaded,hi,hx i.1.val hi,hd i.1.val hi]
  · simp [masked,L2NormTriton1.Tile.masked,hi]

theorem scalar_congr {α : Type} (M : Algebra α) (N B : Nat) (eps : α)
    (xs dys ys dzs : Nat → α)
    (hx : ∀ i,i<N → xs i=ys i) (hd : ∀ i,i<N → dys i=dzs i) (i : Nat) (hi : i<N) :
    scalar M N B eps xs dys i=scalar M N B eps ys dzs i := by
  dsimp only [scalar,variance]
  rw [L2NormTriton1.Tile.variance_congr M N B xs ys hx,dot_congr M N B xs dys ys dzs hx hd,hx i hi,hd i hi]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Tile
