import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridAddresses
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardGridAddresses
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.PreGridAddresses
import bench.repairs.tritonbench_g.chunk_retention.Proofs.PureAccessSafety

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]

/-- Capacities for the actual finite source windows, including the saved
forward/preprocess buffers. No numerical or execution premise occurs here. -/
def Bounds (c : Shape) (bounds : RegionName→Nat) : Prop :=
  (∀ r∈(["Q","K","V","DO","Out","DQ","DK","DV","NewDO"] : List RegionName),c.elements≤bounds r) ∧
  (∀ r∈(["L","M","Delta"] : List RegionName),c.groups*c.rows≤bounds r)

theorem element_bound (c : Shape) (h row d : Nat)
    (hh : h<c.groups) (hr : row<c.rows) (hd : d<c.dim) :
    (h*c.rows+row)*c.dim+d<c.elements := offset_lt c (⟨h,hh⟩,⟨row,hr⟩,⟨d,hd⟩)

theorem row_bound (c : Shape) (h row : Nat) (hh : h<c.groups) (hr : row<c.rows) :
    h*c.rows+row<c.groups*c.rows := rowOffset_lt c ⟨h,hh⟩ ⟨row,hr⟩

theorem tile_row_bound (c : Shape) (block i : Nat) (hb : block<c.blocks) (hi : i<128) :
    block*128+i<c.rows := by
  change _<c.blocks*128
  omega

theorem forward_query_bound (c : Shape) (scale : ℝ) (s : State α)
    (legal : s.pids 0<c.blocks ∧ s.pids 1<c.groups)
    (i : FQuery (forwardParameters c scale)) :
    forwardOffset (forwardParameters c scale) s i<c.elements := by
  have address : forwardOffset (forwardParameters c scale) s i=
      (s.pids 1*c.rows+(s.pids 0*128+i.1.val))*c.dim+i.2.1.val := by
    simp [forwardOffset,forwardHeadRow,forwardQueryRow,forwardParameters,c.dim_pos]
  rw [address]
  exact element_bound c _ _ _ legal.2 (tile_row_bound c _ _ legal.1 i.1.isLt) i.2.1.isLt

theorem forward_kv_bound (c : Shape) (scale : ℝ) (s : State α)
    (legal : s.pids 0<c.blocks ∧ s.pids 1<c.groups) (block : Nat)
    (hb : block<forwardBlocks (forwardParameters c scale) s)
    (i : FKey (forwardParameters c scale)) :
    forwardKVOffset (forwardParameters c scale) s block i<c.elements := by
  have stop : forwardBlocks (forwardParameters c scale) s=s.pids 0+1 := by
    simp [forwardBlocks,forwardStop,forwardParameters]
  have row : block*128+i.1.val<c.rows := tile_row_bound c _ _ (by omega) i.1.isLt
  have address : forwardKVOffset (forwardParameters c scale) s block i=
      (s.pids 1*c.rows+(block*128+i.1.val))*c.dim+i.2.1.val := by
    simp [forwardKVOffset,forwardHeadRow,forwardParameters,c.dim_pos,Nat.add_assoc]
  rw [address]
  exact element_bound c _ _ _ legal.2 row i.2.1.isLt

theorem forward_row_bound (c : Shape) (scale : ℝ) (s : State α)
    (legal : s.pids 0<c.blocks ∧ s.pids 1<c.groups)
    (i : FRow (forwardParameters c scale)) :
    forwardRowOffset (forwardParameters c scale) s i<c.groups*c.rows :=
  row_bound c _ _ legal.2 (tile_row_bound c _ _ legal.1 i.1.isLt)

end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
