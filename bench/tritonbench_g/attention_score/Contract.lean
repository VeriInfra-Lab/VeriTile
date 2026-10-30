import bench.tritonbench_g.attention_score.Proofs.Tensor

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked
open VeriTile Triton Reused Geometry Physical
noncomputable section

/-- Q and K contain B*D elements. The existing B<=64 contract bounds B*B,
B, broadcast vectors and scalars below the same frontend limit. -/
def StaticTiles (c : Shape) : Prop := c.B*c.D ≤ 1048576

def NonemptyLaunch (c : Shape) : Prop := c.N ≠ 0 ∧ c.outputSize ≠ 0

def InputContract (c : Shape) (t : Tensors c) (offset size : Int) (window : Bool) : Prop :=
  Reused.Contract.InputContract c (observations t) offset size window ∧
  (NonemptyLaunch c → StaticTiles c)

def check (c : Shape) (t : Tensors c) (offset size : Int) (window : Bool) : Bool := by
  classical
  exact decide (InputContract c t offset size window)

theorem check_iff (c : Shape) (t : Tensors c) (offset size : Int) (window : Bool) :
    check c t offset size window = Bool.true ↔ InputContract c t offset size window := by
  classical
  simp [check]

/-- Complete distinct tensor shapes formed by the source, including both
block-pointer tiles, score/mask/probability tiles, expanded vectors and scalars. -/
def sourceTileShapes (c : Shape) : List (List Nat) :=
  [[],[c.B],[c.D],[c.B,1],[1,c.B],[c.B,c.D],[c.D,c.B],[c.B,c.B]]

theorem source_tiles_bounded (c : Shape) (h : StaticTiles c) :
    ∀ shape ∈ sourceTileShapes c, shape.prod ≤ 1048576 := by
  have hp := supported_positive c.B c.block
  have hd : c.D ≤ c.B*c.D := by nlinarith
  have hb : c.B ≤ 64 := by rcases c.block with h|h|h <;> omega
  have hs : c.B*c.B ≤ 4096 := by nlinarith
  intro shape hi
  simp only [sourceTileShapes,List.mem_cons,List.not_mem_nil,or_false] at hi
  rcases hi with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one,Nat.one_mul]
  all_goals unfold StaticTiles at h
  all_goals first | exact h | simpa [Nat.mul_comm] using h | omega

theorem empty_preserves_domain (c : Shape) (t : Tensors c) (offset size : Int) (window : Bool)
    (empty : c.N=0 ∨ c.outputSize=0) :
    InputContract c t offset size window ↔ Reused.Contract.InputContract c (observations t) offset size window := by
  simp only [InputContract,NonemptyLaunch]
  constructor
  · exact And.left
  · intro h
    exact ⟨h,fun hn => False.elim (empty.elim hn.1 hn.2)⟩

#specNonCircular InputContract avoiding [FP.Structural.exec]
#axiomsClean check_iff
#axiomsClean source_tiles_bounded
#axiomsClean empty_preserves_domain
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked
