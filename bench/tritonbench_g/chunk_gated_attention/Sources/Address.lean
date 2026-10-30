import bench.tritonbench_g.chunk_gated_attention.Sources.AddressArithmetic
import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHPointers

/-! Rebased pointer observations and signed-64 address refinement. The allocation
extent bound is the ordinary representability property of an allocated tensor,
not a newly imposed native-safe restriction on admitted input dimensions. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Address
open VeriTile.Triton
noncomputable section

/-- Full original matrix pointer before the repair. -/
def parent2 (r : RegionName) (base rows cols br bc sr sc row col : Nat) : BlockPtr :=
  ⟨r,base,[rows,cols],[br,bc],[sr,sc],[row,col]⟩
/-- Actual repaired pointer: the coordinate offset is in the wide base. -/
def local2 (r : RegionName) (base br bc sr sc : Nat) : BlockPtr :=
  ⟨r,base,[br,bc],[br,bc],[sr,sc],[0,0]⟩

theorem rebase2_address (r : RegionName) (base rows cols br bc sr sc row col i j : Nat) :
    (local2 r (base+row*sr+col*sc) br bc sr sc).address [i,j] =
      (parent2 r base rows cols br bc sr sc row col).address [i,j] := by
  simp only [local2,parent2,BlockPtr.address_2d_offsets,Nat.zero_add]
  ring

theorem rebase2_bounds (r : RegionName) (base rows cols br bc sr sc row col : Nat)
    (hr : row+br ≤ rows) (hc : col+bc ≤ cols) (i : Fin br) (j : Fin bc) :
    (local2 r (base+row*sr+col*sc) br bc sr sc).inBounds [i.val,j.val] [0,1] = Bool.true ∧
    (parent2 r base rows cols br bc sr sc row col).inBounds [i.val,j.val] [0,1] = Bool.true := by
  simp only [local2,parent2,BlockPtr.inBounds_2d_offsets,Nat.zero_add,decide_eq_true_eq]
  exact ⟨⟨i.isLt,j.isLt⟩,⟨by omega,by omega⟩⟩

/-- Same checked read for every typed memory cell and every oracle. Thus no
zero-oracle assumption is needed at any repaired fully staged matrix access. -/
theorem rebase2_read (r : RegionName) (base rows cols br bc sr sc row col : Nat)
    (hr : row+br ≤ rows) (hc : col+bc ≤ cols) (i : Fin br) (j : Fin bc)
    {V : Type} (memory padding : RegionName → Nat → V) :
    (let p := local2 r (base+row*sr+col*sc) br bc sr sc
     if p.inBounds [i.val,j.val] [0,1] then memory p.region (p.address [i.val,j.val])
     else padding p.region (p.address [i.val,j.val])) =
    (let p := parent2 r base rows cols br bc sr sc row col
     if p.inBounds [i.val,j.val] [0,1] then memory p.region (p.address [i.val,j.val])
     else padding p.region (p.address [i.val,j.val])) := by
  dsimp only
  rw [(rebase2_bounds r base rows cols br bc sr sc row col hr hc i j).1,
      (rebase2_bounds r base rows cols br bc sr sc row col hr hc i j).2]
  exact congrArg (memory r) (rebase2_address r base rows cols br bc sr sc row col i.val j.val)

/-- Stores hit the identical cell and enable bit; the whole tagged value is
preserved. This is independent of real erasure and floating precision. -/
theorem rebase2_store_observation (r : RegionName) (base rows cols br bc sr sc row col : Nat)
    (hr : row+br ≤ rows) (hc : col+bc ≤ cols) (i : Fin br) (j : Fin bc) :
    ((local2 r (base+row*sr+col*sc) br bc sr sc).region,
      (local2 r (base+row*sr+col*sc) br bc sr sc).address [i.val,j.val],
      (local2 r (base+row*sr+col*sc) br bc sr sc).inBounds [i.val,j.val] [0,1]) =
    ((parent2 r base rows cols br bc sr sc row col).region,
      (parent2 r base rows cols br bc sr sc row col).address [i.val,j.val],
      (parent2 r base rows cols br bc sr sc row col).inBounds [i.val,j.val] [0,1]) := by
  rw [rebase2_address,(rebase2_bounds r base rows cols br bc sr sc row col hr hc i j).1,
      (rebase2_bounds r base rows cols br bc sr sc row col hr hc i j).2]
  rfl

def parent1 (r : RegionName) (base size block stride off : Nat) : BlockPtr :=
  ⟨r,base,[size],[block],[stride],[off]⟩
def local1 (r : RegionName) (base block stride : Nat) : BlockPtr :=
  ⟨r,base,[block],[block],[stride],[0]⟩

theorem rebase1_address (r : RegionName) (base size block stride off i : Nat) :
    (local1 r (base+off*stride) block stride).address [i] =
      (parent1 r base size block stride off).address [i] := by
  simp only [local1,parent1,BlockPtr.address_1d,Nat.zero_add]
  ring

theorem rebase1_bounds (r : RegionName) (base size block stride off : Nat)
    (h : off+block ≤ size) (i : Fin block) :
    (local1 r (base+off*stride) block stride).inBounds [i.val] [0] = Bool.true ∧
    (parent1 r base size block stride off).inBounds [i.val] [0] = Bool.true := by
  simp only [local1,parent1,BlockPtr.inBounds_1d,Nat.zero_add,decide_eq_true_eq]
  exact ⟨i.isLt,by omega⟩

theorem fixed_history_counterexample :
    (historyWord 0 128 4096 4096 257 0 0).toInt = 2147483648 := by
  rw [history_word_eq]
  norm_num [word,BitVec.toInt]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Address
