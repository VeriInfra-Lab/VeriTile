import bench.repairs.tritonbench_g.chunk_cumsum_vector.NativeIO
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Rewrite
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem active_time (h : Call) (c : Config) (p : Geometry.Program h) (q i j : Nat)
    (active : Geometry.active h c p.feature q i j) : q*c.BT+i<h.T := by
  have bound := active.1
  unfold Geometry.activeTime at bound
  omega

/-- Logical FP32-promoted block values, including the source's explicit zero
padding. The definition depends only on input values and source geometry. -/
def loadedInput (M : Algebra α) (h : Call) (c : Config) (p : Geometry.Program h)
    (xs : Values α h.inputDType.toTileDType [h.T,BS]) (q : Nat) : Values α .real [c.BT,BS] :=
  fun i => M.cast (some .fp32) h.inputDType .real
    (if active : Geometry.active h c p.feature q i.1.val i.2.1.val then
      toFloat h.inputDType (xs (⟨q*c.BT+i.1.val,active_time h c p q _ _ active⟩,i.2))
    else M.cast (some .fp32) .real h.inputDType (M.literal (some .fp32) .real 0))

def carryInput (M : Algebra α) (h : Call) (c : Config) (p : Geometry.Program h)
    (xs : Values α h.inputDType.toTileDType [h.T,BS]) : Nat → Values α .real [BS]
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | q+1 => updated M c (loadedInput M h c p xs q) (carryInput M h c p xs q)

/-- Exactly the original ADD-COMMUTE rewrite sites: active feature lanes,
only updates whose carry is consumed by a later output block. -/
def finiteCarry (M : Algebra α) (D : FP.Guarded.Domain α)
    (h : Call) (c : Config) (p : Geometry.Program h)
    (xs : Values α h.inputDType.toTileDType [h.T,BS]) : Prop :=
  ∀ q, q+1<Geometry.chunks h c → ∀ j : TileIndex [BS], p.feature*BS+j.1.val<h.S →
    D .finite (carryInput M h c p xs q j) ∧
    D .finite (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false (loadedInput M h c p xs q) j)

end
end VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
