import bench.tritonbench_g.chunk_linear_attn.IOLayout
import VeriTile.Triton.Core.Reshape

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOArray
open VeriTile Triton
noncomputable section

/-- Row-major finite-array indexing, independent of execution and memory. -/
def index : (shape : TileShape) → Fin (TileShape.product shape) → TileIndex shape
  | [],_ => PUnit.unit
  | n::rest,o => by
    have positive : 0 < TileShape.product rest := by
      have inside := o.isLt
      change o.val < n*TileShape.product rest at inside
      by_contra h
      have zero : TileShape.product rest=0 := by omega
      rw [zero,Nat.mul_zero] at inside
      omega
    exact (⟨o.val/TileShape.product rest,
      (Nat.div_lt_iff_lt_mul positive).mpr (by simpa [Nat.mul_comm] using o.isLt)⟩,
      index rest ⟨o.val%TileShape.product rest,Nat.mod_lt _ positive⟩)

theorem index_offset (shape : TileShape) (o : Fin (TileShape.product shape)) :
    TileShape.linearIndex shape (index shape o) = o.val := by
  induction shape with
  | nil => simp [TileShape.linearIndex]
  | cons n rest ih =>
    simp only [index,TileShape.linearIndex,ih]
    simpa only [Nat.mul_comm] using Nat.div_add_mod o.val (TileShape.product rest)

theorem offset_index (shape : TileShape) (j : TileIndex shape) :
    index shape ⟨TileShape.linearIndex shape j,TileShape.linearIndex_lt shape j⟩ = j := by
  induction shape with
  | nil => exact Subsingleton.elim _ _
  | cons n rest ih =>
    have bound := TileShape.linearIndex_lt rest j.2
    have positive : 0<TileShape.product rest := Nat.zero_lt_of_lt bound
    have quotient : (j.1.val*TileShape.product rest+TileShape.linearIndex rest j.2)/TileShape.product rest=j.1.val := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt bound,Nat.zero_add]
    have modulus : (j.1.val*TileShape.product rest+TileShape.linearIndex rest j.2)%TileShape.product rest=
        TileShape.linearIndex rest j.2 := by
      simp [Nat.mod_eq_of_lt bound]
    apply Prod.ext
    · exact Fin.ext quotient
    · convert ih j.2 using 1
      exact congrArg (index rest) (Fin.ext modulus)

def extend {α : Type} [Inhabited α] (shape : TileShape) (values : TileIndex shape → α) (offset : Nat) : α :=
  if inside : offset < TileShape.product shape then values (index shape ⟨offset,inside⟩) else default

theorem read {α : Type} [Inhabited α] (shape : TileShape) (values : TileIndex shape → α) (j : TileIndex shape) :
    extend shape values (TileShape.linearIndex shape j) = values j := by
  rw [extend,dif_pos (TileShape.linearIndex_lt shape j),offset_index]

theorem capacity (shape : TileShape) (bound : Nat)
    (all : ∀ j : TileIndex shape,TileShape.linearIndex shape j<bound) :
    TileShape.product shape ≤ bound := by
  by_contra small
  have inside : bound<TileShape.product shape := by omega
  have impossible := all (index shape ⟨bound,inside⟩)
  rw [index_offset] at impossible
  exact Nat.lt_irrefl bound impossible

theorem offset3 (a : Fin A) (b : Fin B) (c : Fin C) :
    TileShape.linearIndex [A,B,C] (a,b,c,PUnit.unit) = (a.val*B+b.val)*C+c.val := by
  simp only [TileShape.linearIndex,TileShape.product,Nat.mul_one,Nat.add_zero]
  ring

#axiomsClean read
#axiomsClean capacity
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOArray
