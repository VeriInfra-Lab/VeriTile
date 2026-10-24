import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.PhysicalGrid
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.CallerViews
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.HostCoordinates

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.FPHost
open VeriTile.Triton FP.Structural FPValues Snapshot
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def data (h : Call) (logical : Logical α) (zero : α) (packed : Packed α h logical zero)
    (other : RegionName → Nat → Cell α)
    (padding : (dtype : TileDType) → RegionName → Nat → Value α dtype) : Data α where
  input := packed.input
  weights := packed.weights
  output := packed.output
  indices := packed.indices
  other := other
  padding := padding

def inputLane (M : Algebra α) (h : Call) (logical : Logical α) (zero : α) (batch : Nat) (k : Fin h.BK) : α :=
  M.cast (some .fp32) .real .fp16 (M.fp32Load
    (if k.val<h.K then logical.input batch k.val
     else if h.even then zero else M.literal (some .fp32) .real 0))

def weightLane (M : Algebra α) (h : Call) (logical : Logical α) (slot row : Nat) (n : Fin 256) (k : Fin h.BK) : α :=
  if n.val<1 ∧ k.val<h.K then logical.weight slot row k.val
  else M.cast (some .fp32) .real .fp16 (M.literal (some .fp32) .real 0.0)

def productTile (M : Algebra α) (h : Call) (logical : Logical α) (zero : α)
    (batch slot row : Nat) : Values α .fp16 [256,h.BK] :=
  fun (n,k,_) => M.binary (some .fp32) .fp16 .mul
    (inputLane M h logical zero batch k) (weightLane M h logical slot row n k)

/-- Independent caller-coordinate source-order numerical reference. The local
FP16 reduction sees its full original 256×BK tile, with exact source masks and
literals; it is not replaced by a full-output oracle or a Real sum. -/
def reference (M : Algebra α) (reduction : Reduction α) (h : Call) (logical : Logical α)
    (zero : α) (batch slot row : Nat) : α :=
  let total := M.cast (some .fp32) .fp16 .real
    (reduction.apply (some .fp32) .fp16 ⟨1,by simp⟩ Bool.false
      (productTile M h logical zero batch slot row) (⟨0,by decide⟩,PUnit.unit))
  if h.add then M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add total (M.fp32Load (logical.output batch (h.slice+row)))) else total

theorem loaded_snapshot (M : Algebra α) (h : Call) (logical : Logical α) (zero : α)
    (packed : Packed α h logical zero) (other : RegionName → Nat → Cell α)
    (padding : (dtype : TileDType) → RegionName → Nat → Value α dtype)
    (p : Geometry.Program h) (k : Fin h.BK) :
    loaded M h p.batch (data h logical zero packed other padding) k=inputLane M h logical zero p.batch k := by
  by_cases inside : k.val<h.K
  · simp [loaded,inputLane,data,inside,packed.inputCopy p.batch k.val p.batchInside inside]
  · have above : h.K≤k.val := by omega
    cases even : h.even <;>
      simp [loaded,inputLane,data,inside,even,packed.inputZero p.batch k.val p.batchInside above k.isLt]

theorem weight_snapshot (M : Algebra α) (h : Call) (logical : Logical α) (zero : α)
    (packed : Packed α h logical zero) (other : RegionName → Nat → Cell α)
    (padding : (dtype : TileDType) → RegionName → Nat → Value α dtype)
    (p : Geometry.Program h) (slot : Nat) (selected : slot<h.adapters) (n : Fin 256) (k : Fin h.BK) :
    weight M h slot p.row (data h logical zero packed other padding) n k=weightLane M h logical slot p.row n k := by
  by_cases active : n.val<1 ∧ k.val<h.K
  · have lane : n.val=0 := by omega
    simp [weight,weightLane,active,data,lane,packed.weightCopy slot p.row k.val selected p.rowInside active.2]
  · simp only [weight,weightLane,active,if_false]

theorem value_snapshot (M : Algebra α) (reduction : Reduction α) (h : Call)
    (logical : Logical α) (zero : α) (packed : Packed α h logical zero)
    (other : RegionName → Nat → Cell α)
    (padding : (dtype : TileDType) → RegionName → Nat → Value α dtype)
    (p : Geometry.Program h) (selected : (logical.indices p.batch).toNat<h.adapters) :
    FPGrid.value M reduction h p (data h logical zero packed other padding)=
      reference M reduction h logical zero p.batch (logical.indices p.batch).toNat p.row := by
  have products : FPValues.products M h p.batch (logical.indices p.batch).toNat p.row
      (data h logical zero packed other padding)=
      productTile M h logical zero p.batch (logical.indices p.batch).toNat p.row := by
    funext i
    rcases i with ⟨n,k,unit⟩
    simp only [FPValues.products,productTile,loaded_snapshot M h logical zero packed other padding p k,
      weight_snapshot M h logical zero packed other padding p _ selected n k]
  have column : h.slice+p.row<h.columns := by have := p.rowInside; have := h.sliceInside; omega
  have output := packed.outputCopy p.batch (h.slice+p.row) p.batchInside column
  have output' : packed.output (p.batch*(h.columns+255)+p.row+h.slice)=logical.output p.batch (h.slice+p.row) := by
    simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using output
  simp only [FPGrid.value,result,sums,reference]
  rw [show (data h logical zero packed other padding).indices p.batch=logical.indices p.batch from packed.metadataCopy p.batch p.batchInside,products]
  simp [old,data,output']
  rfl

#axiomsClean value_snapshot
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.FPHost
