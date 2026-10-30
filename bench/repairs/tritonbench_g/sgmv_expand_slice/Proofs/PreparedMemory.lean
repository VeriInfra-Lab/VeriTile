import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridSchedule
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SnapshotLayout
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Metadata

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PreparedMemory
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 20000

structure Strides where
  (inputRow inputInner weightAdapter weightColumn weightInner outputRow outputColumn : Nat)

def arguments (shape : Shape) (strides : Strides) : KernelArgs :=
  ⟨"sgmv.private.input","sgmv.private.weight","sgmv.private.output",
    "sgmv.private.starts","sgmv.private.lengths","sgmv.private.adapters",
    shape.columns,shape.inner,strides.inputRow,strides.inputInner,strides.weightAdapter,
    strides.weightColumn,strides.weightInner,strides.outputRow,strides.outputColumn,shape.slice,0⟩

def state {α : Type} [Inhabited α] (abi : SourceABI) (shape : Shape)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (Y : Nat → Nat → α) (raw : List RawSegment) (extraLengths : List Int) : State α where
  mem := fun region address =>
    if region = "sgmv.private.input" then
      .mk abi.input.dtype.toTileDType (ofFloat abi.input.dtype (X (address/shape.inner) (address%shape.inner)))
    else if region = "sgmv.private.weight" then
      .mk abi.weight.scalar.dtype.toTileDType (ofFloat abi.weight.scalar.dtype
        (W (address/shape.inner/shape.columns) (address/shape.inner%shape.columns) (address%shape.inner)))
    else if region = "sgmv.private.output" then
      .mk (SourceSuffix.storageType abi.output).toTileDType (ofFloat (SourceSuffix.storageType abi.output)
        (Y (address/shape.outputColumns) (address%shape.outputColumns)))
    else if region = "sgmv.private.starts" then .mk .int ((raw[address]?).getD ⟨0,0,-1⟩).start
    else if region = "sgmv.private.lengths" then .mk .int ((((raw.map RawSegment.length)++extraLengths)[address]?).getD 0)
    else if region = "sgmv.private.adapters" then .mk .int ((raw[address]?).getD ⟨0,0,-1⟩).adapter
    else .mk .int 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

@[simp] theorem float_roundtrip {α : Type} (d : FloatDType) (value : α) :
    toFloat d (ofFloat d value) = value := by cases d <;> rfl

theorem matrix_row (r c columns : Nat) (hc : c < columns) : (r*columns+c)/columns = r := by
  have hp : 0 < columns := by omega
  simpa [Nat.add_comm,Nat.div_eq_of_lt hc] using Nat.add_mul_div_right c r hp

theorem matrix_column (r c columns : Nat) (hc : c < columns) : (r*columns+c)%columns = c := by
  simp [Nat.add_mod,Nat.mod_eq_of_lt hc]

theorem private_regions (shape : Shape) (strides : Strides) : SourceGrid.PrivateRegions (arguments shape strides) := by
  constructor <;> simp [arguments]

theorem inputs {α : Type} [Inhabited α] (abi : SourceABI) (shape : Shape) (strides : Strides)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (Y : Nat → Nat → α) (raw : List RawSegment) (extraLengths : List Int)
    (xlayout : GridStride.Contiguous shape.tokens shape.inner strides.inputRow strides.inputInner)
    (wlayout : SnapshotLayout.WeightContiguous shape.adapters shape.columns shape.inner
      strides.weightAdapter strides.weightColumn strides.weightInner) :
    SourceMemory.Inputs abi (arguments shape strides) shape (state abi shape X W Y raw extraLengths) X W := by
  refine ⟨rfl,rfl,?_,?_⟩
  · intro r k hr hk
    change toFloat abi.input.dtype ((state abi shape X W Y raw extraLengths).mem "sgmv.private.input"
      (r*strides.inputRow+k*strides.inputInner) |>.read abi.input.dtype.toTileDType) = _
    rw [GridStride.coordinate_linear xlayout hr hk]
    simp [state,matrix_row r k shape.inner hk,matrix_column r k shape.inner hk,
      float_roundtrip]
  · intro a c k ha hc hk
    change toFloat abi.weight.scalar.dtype ((state abi shape X W Y raw extraLengths).mem "sgmv.private.weight"
      (strides.weightAdapter*a+k*strides.weightInner+c*strides.weightColumn) |>.read abi.weight.scalar.dtype.toTileDType) = _
    rw [SnapshotLayout.weight_linear wlayout ha hc hk]
    simp [state,matrix_row _ k shape.inner hk,matrix_column _ k shape.inner hk,
      matrix_row a c shape.columns hc,matrix_column a c shape.columns hc,float_roundtrip]

theorem output {α : Type} [Inhabited α] (abi : SourceABI) (shape : Shape) (strides : Strides)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (Y : Nat → Nat → α) (raw : List RawSegment) (extraLengths : List Int)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns strides.outputRow strides.outputColumn) :
    SourceSuffix.OutputMemory abi (arguments shape strides) shape (state abi shape X W Y raw extraLengths) Y := by
  intro r c hr hc
  unfold SourceSuffix.readOutput
  change toFloat (SourceSuffix.storageType abi.output) ((state abi shape X W Y raw extraLengths).mem "sgmv.private.output"
    (r*strides.outputRow+c*strides.outputColumn) |>.read (SourceSuffix.storageType abi.output).toTileDType) = _
  rw [GridStride.coordinate_linear layout hr hc]
  simp [state,matrix_row r c shape.outputColumns hc,matrix_column r c shape.outputColumns hc,
    float_roundtrip]

#axiomsClean private_regions
#axiomsClean inputs
#axiomsClean output
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PreparedMemory
