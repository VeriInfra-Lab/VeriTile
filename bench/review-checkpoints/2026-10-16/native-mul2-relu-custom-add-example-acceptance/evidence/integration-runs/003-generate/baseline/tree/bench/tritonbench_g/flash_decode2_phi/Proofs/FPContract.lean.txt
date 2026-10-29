import bench.tritonbench_g.flash_decode2_phi.Proofs.FPSource
import bench.tritonbench_g.flash_decode2_llama.Proofs.FPSoftware

/-! Independent input-memory formulas for the accepted Phi stages. All scalar
operations remain distinct backend calls; inactive lanes retain source padding
until the masked store. No reassociation or zero-annihilation law is used. -/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
open VeriTile.Triton FP.Structural
open FlashDecode2Llama.FPStages (Row Scalar address scalarAddress ScalarPost scalar_post)
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure OutputParams where
  headDim : Nat
  width : Nat
  input : Row
  sum : Scalar
  output : Row

structure StepParams where
  block : Nat
  headDim : Nat
  width : Nat
  mid : Row
  midBlockStride : Nat
  log : Scalar
  acc : Row
  sum : Scalar
  logic : Scalar

def normalization (Acc SumExp Out : RegionName) (p : OutputParams) : ComputeKernel :=
  source_normalization_store_kernel Acc SumExp Out p.headDim
    p.input.batchStride p.input.headStride p.input.laneStride p.sum.batchStride p.sum.headStride
    p.output.batchStride p.output.headStride p.output.laneStride p.width

def copy (Final Out : RegionName) (p : OutputParams) : ComputeKernel :=
  source_final_store_slice Final Out p.headDim p.input.batchStride p.input.headStride p.input.laneStride
    p.output.batchStride p.output.headStride p.output.laneStride p.width

def accumulator (Mid Log AccIn Prev Next AccOut : RegionName) (p : StepParams) : ComputeKernel :=
  source_accumulator_step_kernel Mid Log AccIn Prev Next AccOut p.block p.headDim
    p.mid.batchStride p.mid.headStride p.midBlockStride p.mid.laneStride p.log.batchStride p.log.headStride
    p.acc.batchStride p.acc.headStride p.acc.laneStride p.logic.batchStride p.logic.headStride p.width

def sumStep (Log SumIn Prev Next SumOut : RegionName) (p : StepParams) : ComputeKernel :=
  source_sum_exp_step_kernel Log SumIn Prev Next SumOut p.block p.log.batchStride p.log.headStride
    p.sum.batchStride p.sum.headStride p.logic.batchStride p.logic.headStride

def read {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName) (o : Nat) : α :=
  M.fp32Load ((s.mem R o).read .real)

def maskedRead {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (o : Nat) (active : Prop) [Decidable active] : α :=
  M.fp32Load (if active then (s.mem R o).read .real else M.literal (some .fp32) .real 0)

def normalizationValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Acc SumExp : RegionName) (p : OutputParams) : TileIndex [p.width] → α := fun i =>
  M.binary (some .fp32) .real .div
    (maskedRead M s Acc (address s p.input i) (i.1.val < p.headDim))
    (read M s SumExp (scalarAddress s p.sum))

def copyValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Final : RegionName) (p : OutputParams) : TileIndex [p.width] → α := fun i =>
  maskedRead M s Final (address s p.input i) (i.1.val < p.headDim)

def midAddress {α : Type} (s : State α) (p : StepParams) (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.mid.batchStride + s.pids 1 * p.mid.headStride +
    p.block * p.midBlockStride + i.1.val * p.mid.laneStride

def oldScale {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Prev Next : RegionName) (p : StepParams) : α :=
  M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub
    (read M s Prev (scalarAddress s p.logic)) (read M s Next (scalarAddress s p.logic)))

def expLogic {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Log Next : RegionName) (p : StepParams) : α :=
  M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub
    (read M s Log (scalarAddress s p.log + p.block)) (read M s Next (scalarAddress s p.logic)))

def accumulatorValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Mid Log AccIn Prev Next : RegionName) (p : StepParams) : TileIndex [p.width] → α := fun i =>
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (maskedRead M s AccIn (address s p.acc i) (i.1.val < p.headDim)) (oldScale M s Prev Next p))
    (M.binary (some .fp32) .real .mul (expLogic M s Log Next p)
      (maskedRead M s Mid (midAddress s p i) (i.1.val < p.headDim)))

def sumValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Log SumIn Prev Next : RegionName) (p : StepParams) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (read M s SumIn (scalarAddress s p.sum)) (oldScale M s Prev Next p))
    (expLogic M s Log Next p)

def Post {α : Type} (s t : State α) (Out : RegionName) (o : Row) (headDim D : Nat)
    (values : TileIndex [D] → α) : Prop :=
  (∀ i, i.1.val < headDim → t.mem Out (address s o i) = .mk .real (values i)) ∧
  (∀ R off, (R ≠ Out ∨ ∀ i : TileIndex [D], i.1.val < headDim → off ≠ address s o i) →
    t.mem R off = s.mem R off)

theorem post_of_memory {α : Type} (s t : State α) (Out : RegionName) (o : Row) (headDim D : Nat)
    (v : TileIndex [D] → α) (hinj : Function.Injective (address s o (D := D)))
    (hm : t.mem = (scatter s Out D (address s o) (fun i => i.1.val < headDim) v).mem) :
    Post s t Out o headDim D v := by
  constructor
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback Out (address s o) (fun j => .mk .real (v j))
      (fun j => j.1.val < headDim) s i hi hinj
  · intro R off h
    rw [hm]
    apply State.masked_scatter_frame
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => h i hi

theorem normalization_memory {α : Type} [Inhabited α] (M : Algebra α)
    (Acc SumExp Out : RegionName) (p : OutputParams) (s : State α) :
    ∃ t, FP.Structural.exec M (normalization Acc SumExp Out p) s = some t ∧
      t.mem = (scatter s Out p.width (address s p.output) (fun i => i.1.val < p.headDim)
        (normalizationValues M s Acc SumExp p)).mem := by
  simp [normalization, source_normalization_store_kernel, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    Region.cast, normalizationValues, maskedRead, read, address, scalarAddress, scatter,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem copy_memory {α : Type} [Inhabited α] (M : Algebra α)
    (Final Out : RegionName) (p : OutputParams) (s : State α) :
    ∃ t, FP.Structural.exec M (copy Final Out p) s = some t ∧
      t.mem = (scatter s Out p.width (address s p.output) (fun i => i.1.val < p.headDim)
        (copyValues M s Final p)).mem := by
  simp [copy, source_final_store_slice, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    Region.cast, copyValues, maskedRead, address, scatter,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem accumulator_memory {α : Type} [Inhabited α] (M : Algebra α)
    (Mid Log AccIn Prev Next AccOut : RegionName) (p : StepParams) (s : State α) :
    ∃ t, FP.Structural.exec M (accumulator Mid Log AccIn Prev Next AccOut p) s = some t ∧
      t.mem = (scatter s AccOut p.width (address s p.acc) (fun i => i.1.val < p.headDim)
        (accumulatorValues M s Mid Log AccIn Prev Next p)).mem := by
  simp [accumulator, source_accumulator_step_kernel, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    Region.cast, accumulatorValues, oldScale, expLogic, maskedRead, read,
    address, midAddress, scalarAddress, scatter, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem sum_memory {α : Type} [Inhabited α] (M : Algebra α)
    (Log SumIn Prev Next SumOut : RegionName) (p : StepParams) (s : State α) :
    ∃ t, FP.Structural.exec M (sumStep Log SumIn Prev Next SumOut p) s = some t ∧
      t.mem = (s.write SumOut (scalarAddress s p.sum) (.mk .real (sumValue M s Log SumIn Prev Next p))).mem := by
  simp [sumStep, source_sum_exp_step_kernel, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    Region.cast, sumValue, oldScale, expLogic, read, scalarAddress, State.write, State.setReg]

/-- Floating counterpart of the previously accepted recurrence identity.
This definition is not a complete-loop execution theorem. -/
def runningMaximum {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Log : RegionName) (l : Scalar) : Nat → α
  | 0 => M.negInf
  | n+1 => M.binary (some .fp32) .real .max
      (read M s Log (scalarAddress s l + n)) (runningMaximum M s Log l n)

#axiomsClean normalization_memory
#axiomsClean copy_memory
#axiomsClean accumulator_memory
#axiomsClean sum_memory
#axiomsClean post_of_memory
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
