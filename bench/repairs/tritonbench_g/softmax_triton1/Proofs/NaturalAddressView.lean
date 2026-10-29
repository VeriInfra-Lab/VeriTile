import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Real
import VeriTile.Triton.Memory.KernelMemoryTransport

/-! Internal natural-address proof view. Canonical source remains the unchanged
pasted Python body in Kernels.lean. Complete flat and regional memory executions
connect this auxiliary representation to that source, including failure. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton1.NaturalAddressView
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def softmax_kernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + $(input_offset) + row_idx * $(input_row_stride)
    col_offsets = tl.arange(0, $(BLOCK_SIZE))
    input_ptrs = row_start_ptr + col_offsets * $(input_col_stride)
    row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + $(output_offset) + row_idx * $(output_row_stride)
    output_ptrs = output_row_start_ptr + col_offsets * $(output_col_stride)
    tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
  }

def reciprocal_kernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + $(input_offset) + row_idx * $(input_row_stride)
    col_offsets = tl.arange(0, $(BLOCK_SIZE))
    input_ptrs = row_start_ptr + col_offsets * $(input_col_stride)
    row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator * (1 / denominator)
    output_row_start_ptr = output_ptr + $(output_offset) + row_idx * $(output_row_stride)
    output_ptrs = output_row_start_ptr + col_offsets * $(output_col_stride)
    tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
  }

def view (candidate : Bool) (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :=
  if candidate then reciprocal_kernel out inp ib ob ir ic orow oc n B
  else softmax_kernel out inp ib ob ir ic orow oc n B

def source (candidate : Bool) (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :=
  if candidate then SoftmaxTriton1.reciprocal_kernel out inp ib ob ir ic orow oc n B
  else SoftmaxTriton1.softmax_kernel out inp ib ob ir ic orow oc n B

theorem flattenOk (v : Bool) (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :
    (view v out inp ib ob ir ic orow oc n B).toAlgKernel.FlattenOk := by
  cases v <;> simp [view,softmax_kernel,reciprocal_kernel,Kernel.FlattenOk,
    ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

theorem traceSafe (v : Bool) (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0 < B) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ i : Fin B, i.val<n → ib+s.pid*ir+i.val*ic<bounds inp)
    (hout : ∀ i : Fin B, i.val<n → ob+s.pid*orow+i.val*oc<bounds out) :
    Kernel.TraceSafe bounds (view v out inp ib ob ir ic orow oc n B).toAlgKernel s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp only [BlockState.pid_eq] at hin hout
  cases v <;> simp [view,softmax_kernel,reciprocal_kernel,Kernel.TraceSafe,
    ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,
    stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt]
  all_goals exact ⟨fun a ha => hin a ha,fun a ha => hout a ha⟩

private theorem fold_mem {ι : Type} (items : List ι) (o : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (u : BlockState) :
    (items.foldl (fun t i => if active i then t.writeMem o (offset i) (value i) else t) u).mem =
      items.foldl (fun mem i => if active i then
        fun r off => if r=o ∧ off=offset i then MemCell.real (value i) else mem r off
        else mem) u.mem := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i <;> simp only [hi,ite_true,ite_false,ih] <;> rfl

theorem regional_memory (v : Bool) (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0 < B) (s : BlockState) :
    (exec (source v out inp ib ob ir ic orow oc n B).toAlgKernel s).map BlockState.mem =
      (exec (view v out inp ib ob ir ic orow oc n B).toAlgKernel s).map BlockState.mem := by
  cases v <;> simp [view,source,softmax_kernel,reciprocal_kernel,SoftmaxTriton1.softmax_kernel,
      SoftmaxTriton1.reciprocal_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,exec,stepStmts,stepStmt,evalOp.eq_def,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
      Region.cast,hB,←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      Nat.add_assoc,fold_mem]

  all_goals rfl

#axiomsClean regional_memory


end
end VeriTile.Bench.Repairs.SoftmaxTriton1.NaturalAddressView
