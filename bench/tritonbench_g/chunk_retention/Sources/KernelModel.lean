import bench.tritonbench_g.chunk_retention.Sources.Kernels
import bench.tritonbench_g.chunk_retention.Sources.Geometry
import VeriTile.Triton.Float.ProfiledSoundness
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

/-! Local total matrix operation used only while deriving the source program's
execution. Numerical instantiations replace this with an explicit scalar graph;
no kernel, store, recurrence or output port occurs in this interface. -/
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel
open VeriTile Triton FP.Structural
noncomputable section

structure DotPrimitive (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) →
    Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def withDot {α : Type} (M : Algebra α) (D : DotPrimitive α) : Algebra α :=
  { M with
    dot := fun p {_} {_} {_} {_} a b => some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p o {_} {_} {_} {_} a b => some (D.apply p .real .real .real o a b)
    dotTyped := fun p l r o options {_} {_} {_} {_} a b => some (D.apply p l r o options a b) }

def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def rate (M : Algebra α) (c : Shape) (head : Nat) : α :=
  M.unary (some .fp32) .log2
    (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real 1)
      (M.unary (some .fp32) .exp2
        (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real (-5))
          (M.binary (some .fp32) .real .mul
            (M.fromNat (some .fp32) (head % c.H))
            (M.literal (some .fp32) .real 1)))))
def decay (M : Algebra α) (c : Shape) (head : Nat) (n : Int) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .mul (M.fromInt (some .fp32) n) (rate M c head))

/-- The explicit signed32 cast is exact on every temporal lane. -/
@[simp] theorem lane32 (i : Fin 64) :
    (if 2 * (Int.toNat ((i.val:Int) % 4294967296) % 4294967296) < 4294967296 then
      max ((i.val:Int) % 4294967296) 0 % 4294967296 else
      max ((i.val:Int) % 4294967296) 0 % 4294967296 - 4294967296) = (i.val:Int) := by
  have hi := i.isLt
  have hmod : (i.val:Int) % 4294967296 = (i.val:Int) := Int.emod_eq_of_lt (by omega) (by omega)
  simp [hmod,Nat.mod_eq_of_lt (show i.val<4294967296 by omega),
    show 2*i.val<4294967296 by omega,max_eq_left (show (0:Int)≤i.val by omega)]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel
