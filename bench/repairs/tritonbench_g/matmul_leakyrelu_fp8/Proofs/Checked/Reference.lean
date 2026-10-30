import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Bytes
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.FPReference

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- A caller-indexed recurrence: no PID, launch result or final-memory value
occurs. Every source BK64 iteration composes eight fixed K8 result lanes. -/
def referenceAcc (ops : Operations α) (args : Host.Arguments) (memory : Memory α) (row column : Nat) : Nat → α
  | 0 => ops.zero
  | count+1 => block ops
    (fun k => if count*64+k.val < args.A.columns then memory (args.A.address row (count*64+k.val)) else ops.zero)
    (fun k => if count*64+k.val < args.A.columns then memory (args.B.address (count*64+k.val) column) else ops.zero)
    (referenceAcc ops args memory row column count)

def reference (ops : Operations α) (args : Host.Arguments) (memory : Memory α) (row column : Nat) : α :=
  if args.A.columns = 0 then ops.zero
  else ops.castHalf (activate ops args.problem.enabled (referenceAcc ops args memory row column args.problem.kTiles))

theorem accumulator_reference (ops : Operations α) (args : Host.Arguments) (output : Tensor)
    (memory : Memory α) (tile count : Nat) (lane : Source.Lane) (active : Source.active args.problem tile lane) :
    Source.accumulator (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory tile count lane.1 lane.2 =
      referenceAcc ops args memory (args.problem.row tile lane.1.val) (args.problem.column tile lane.2.val) count := by
  have rowBound : args.problem.row tile lane.1.val < args.A.rows := active.1
  have columnBound : args.problem.column tile lane.2.val < args.B.columns := active.2
  have aa (iteration : Nat) :
      Source.loadA (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory tile iteration lane.1 =
      (fun k : Fin 64 => if iteration*64+k.val < args.A.columns then
        memory (args.A.address (args.problem.row tile lane.1.val) (iteration*64+k.val)) else ops.zero) := by
    funext k
    change (if iteration*64+k.val < args.A.columns then memory
      (args.A.address ((args.problem.row tile lane.1.val)%args.A.rows) (clamp args.A.columns (iteration*64+k.val)))
      else ops.zero) = _
    rw [Nat.mod_eq_of_lt rowBound]
    split_ifs with hk
    · simp only [clamp,if_pos hk]
    · rfl
  have bb (iteration : Nat) :
      (fun k => Source.loadB (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory tile iteration k lane.2) =
      (fun k : Fin 64 => if iteration*64+k.val < args.A.columns then
        memory (args.B.address (iteration*64+k.val) (args.problem.column tile lane.2.val)) else ops.zero) := by
    funext k
    change (if iteration*64+k.val < args.A.columns then memory
      (args.B.address (clamp args.A.columns (iteration*64+k.val)) ((args.problem.column tile lane.2.val)%args.B.columns))
      else ops.zero) = _
    rw [Nat.mod_eq_of_lt columnBound]
    split_ifs with hk
    · simp only [clamp,if_pos hk]
    · rfl
  induction count with
  | zero => rfl
  | succ count ih =>
    change block ops (Source.loadA (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory tile count lane.1)
      (fun k => Source.loadB (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory tile count k lane.2)
      (Source.accumulator (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory tile count lane.1 lane.2) =
      block ops _ _ (referenceAcc ops args memory (args.problem.row tile lane.1.val) (args.problem.column tile lane.2.val) count)
    rw [aa count,ih]
    congr 1
    exact bb count

theorem grid_reference (ops : Operations α) (args : Host.Arguments) (output : Tensor)
    (layout : Layout args.problem ⟨args.A,args.B,output⟩) (memory : Memory α)
    (positiveK : args.A.columns ≠ 0) (row column : Nat) (hr : row < args.A.rows) (hc : column < args.B.columns) :
    Source.run (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory (output.address row column) =
      reference ops args memory row column := by
  obtain ⟨tile,ht,lane,active,rows,columns⟩ := Source.output_covered args.problem row column hr hc
  have address : Source.outputAddress args.problem ⟨args.A,args.B,output⟩ tile lane = output.address row column := by
    rw [Source.output_address _ _ _ _ active,rows,columns]
  rw [←address,Source.source_output _ _ _ layout memory tile ht lane active]
  change ops.castHalf (activate ops args.problem.enabled _) = _
  rw [accumulator_reference ops args output memory tile _ lane active,rows,columns,reference,if_neg positiveK]

theorem output_allowed (args : Host.Arguments) (output : Tensor)
    (layout : Layout args.problem ⟨args.A,args.B,output⟩)
    (row column : Nat) (hr : row < args.A.rows) (hc : column < args.B.columns) : Allowed output (output.address row column) := by
  have positive : 1 ≤ args.B.columns := by omega
  have bound := Nat.mul_le_mul_right args.B.columns (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at bound
  have rs : output.rowStride = max args.B.columns 1 := layout.outputStrides.1
  have cs : output.columnStride = 1 := layout.outputStrides.2
  have rows : output.rows = args.A.rows := layout.cShape.1
  have columns : output.columns = args.B.columns := layout.cShape.2
  change output.storage = output.storage ∧ output.offset ≤ output.offset+row*output.rowStride+column*output.columnStride ∧
    output.offset+row*output.rowStride+column*output.columnStride-output.offset < output.rows*output.columns
  rw [rs,cs,rows,columns,max_eq_left positive]
  simp only [Nat.mul_one]
  exact ⟨trivial,by omega,by omega⟩

theorem result_reference (ops : Operations α) (args : Host.Arguments) (output : Tensor)
    (layout : Layout args.problem ⟨args.A,args.B,output⟩) (memory : Memory α)
    (row column : Nat) (hr : row < args.A.rows) (hc : column < args.B.columns) :
    resultMemory ops args output memory (output.address row column) = reference ops args memory row column := by
  have nonempty : ¬(args.A.rows = 0 ∨ args.B.columns = 0) := by omega
  unfold resultMemory
  rw [if_neg nonempty]
  split_ifs with zero
  · have active := output_allowed args output layout row column hr hc
    unfold Allowed at active
    rw [zeroMemory,if_pos active,reference,if_pos zero]
  · exact grid_reference ops args output layout memory zero row column hr hc

theorem reference_stable (ops : Operations α) (args : Host.Arguments) (before after : Memory α)
    (same : ∀ point, point.1 = args.A.storage ∨ point.1 = args.B.storage → after point = before point)
    (row column : Nat) : reference ops args after row column = reference ops args before row column := by
  have recurrence (count : Nat) : referenceAcc ops args after row column count = referenceAcc ops args before row column count := by
    induction count with
    | zero => rfl
    | succ count ih =>
      simp only [referenceAcc,ih]
      congr 1
      · funext k
        split
        · exact same _ (Or.inl rfl)
        · rfl
      · funext k
        split
        · exact same _ (Or.inr rfl)
        · rfl
  simp only [reference,recurrence]


/-- Native operation ports have fixed scalar arity or exactly two K8 operand
fragments and one scalar accumulator. The zero literal is exact, not a port. -/
structure PrimitiveOps where
  castHalf : Word → Word
  nonnegative : Word → Bool
  slopeMultiply : Word → Word
  instruction : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def PrimitiveOps.operations (ops : PrimitiveOps) : Operations Word :=
  ⟨0,ops.castHalf,ops.nonnegative,ops.slopeMultiply,ops.instruction⟩

abbrev InstructionProfile := (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def fpValue (word : Word) : VeriTile.Triton.FP.Value .fp32 := ⟨word⟩

def fpReference (instruction : InstructionProfile) : Operations Word where
  zero := 0
  castHalf word := (TritonLinearActivation.packHalf (fpValue word)).bits
  nonnegative word := let value := fpValue word; !value.isNaN && (!value.sign || value.isZero)
  slopeMultiply word := (VeriTile.Triton.FP.mul VeriTile.Triton.FP.Config.ieeeValue
    (VeriTile.Triton.FP.Value.ofBits .fp32 0x3c23d70a) (fpValue word)).bits
  instruction := instruction

structure LocalPorts (native : PrimitiveOps) (instruction : InstructionProfile) : Prop where
  cast : ∀ x, native.castHalf x = (fpReference instruction).castHalf x
  comparison : ∀ x, native.nonnegative x = (fpReference instruction).nonnegative x
  slope : ∀ x, native.slopeMultiply x = (fpReference instruction).slopeMultiply x
  mma : ∀ xs ys z, native.instruction xs ys z = instruction xs ys z

theorem operations_eq (ports : LocalPorts native instruction) : native.operations = fpReference instruction := by
  have cast := funext ports.cast
  have comparison := funext ports.comparison
  have slope := funext ports.slope
  have mma := funext (fun xs => funext (fun ys => funext (ports.mma xs ys)))
  cases native
  simp_all [PrimitiveOps.operations,fpReference]

#axiomsClean grid_reference
#axiomsClean reference_stable
#axiomsClean operations_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
