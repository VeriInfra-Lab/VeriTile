import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledGraph

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
open Source Reduction
open Scheduled (decode atIndex productInput smoothingInput maskedProduct boundedInput boundedActive)
noncomputable section
set_option maxHeartbeats 10000000

/-- Backend reduction data consists only of explicit scalar instruction
graphs and complete leaf permutations. It cannot return a floating result.
The family is selected from actual launch metadata by the compiler API. -/
structure DevicePlan (width : Nat) where
  maximum : Plan width
  exponentialSum : Plan width
  smoothing : SmoothingProgram
  smoothingCovers : smoothing.leaves.Perm (List.range width)

abbrev DeviceFamily := ∀ width, 0<width → DevicePlan width

def profile (device : DeviceFamily) (host : Profile) : Profile :=
  {host with
    maxTree := fun width h => (device width h).maximum
    sumTree := fun width h => (device width h).exponentialSum}

def emitSum (device : DeviceFamily) (scale : ℝ) {width : Nat} (xs : Fin width → Compiled) : Compiled :=
  if positive : 0<width then
    if smoothingInput (xs ⟨0,positive⟩) then
      ⟨(device width positive).smoothing.emit (.literal scale)
        (fun i => productInput (atIndex xs i)) (fun i => (atIndex xs i).product.isSome),none⟩
    else ⟨.sum width (fun i => (xs i).value),none⟩
  else ⟨.sum width (fun i => (xs i).value),none⟩

def algebra (device : DeviceFamily) (scale : ℝ) : FP32 Compiled :=
  {Scheduled.algebra .pureTrees scale with sum := emitSum device scale}

theorem emit_masked_sum (device : DeviceFamily) (scale : ℝ) {width : Nat} (positive : 0<width)
    (active : Fin width → Bool) (input : Fin width → Instruction) :
    emitSum device scale (fun i => maskedProduct scale (active i) (input i))=
      ⟨(device width positive).smoothing.emit (.literal scale) (boundedInput active input) (boundedActive active),none⟩ := by
  simp only [emitSum,positive,↓reduceDIte,Scheduled.masked_product_tag,↓reduceIte]
  congr 2
  · funext i
    unfold atIndex boundedInput
    split
    · exact Scheduled.masked_product_input _ _ _
    · rfl
  · funext i
    unfold atIndex boundedActive
    split
    · exact Scheduled.masked_product_active _ _ _
    · rfl

theorem exponential_sum (device : DeviceFamily) (scale : ℝ) {width : Nat}
    (xs : Fin width → Instruction) :
    emitSum device scale (fun i => ⟨.unary .exp (xs i),none⟩)=
      ⟨.sum width (fun i => .unary .exp (xs i)),none⟩ := by
  unfold emitSum
  split <;> rfl

theorem source_smoothing_lowering (device : DeviceFamily) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) (smoothing : p.hasSmoothing=Bool.true) :
    (values (algebra device p.logitScale) p memory row block).sumLogits.getD
      ((algebra device p.logitScale).literal 0)=
    emitSum device p.logitScale (fun lane : Fin p.block =>
      maskedProduct p.logitScale
        (decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols))
        ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
          (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value) := by
  have operands := funext (Scheduled.source_smoothing_operands .pureTrees p memory row block)
  simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,smoothing,↓reduceIte,Option.getD]
  exact congrArg (emitSum device p.logitScale) operands

def smoothingValue (device : DeviceFamily) (n : Ports) (profile : Profile) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) : Word :=
  if positive : 0<p.block then
    let active := fun lane : Fin p.block =>
      decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols)
    let input := fun lane : Fin p.block =>
      ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value
    (device p.block positive).smoothing.run n (n.literal p.logitScale)
      (fun i => runInstruction n profile (boundedInput active input i)) (boundedActive active)
  else n.literal 0

theorem source_smoothing_scalar (device : DeviceFamily) (n : Ports) (profile : Profile) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) (positive : 0<p.block) :
    decode n profile ((values (algebra device p.logitScale) p memory row block).sumLogits.getD
      ((algebra device p.logitScale).literal 0))=
      if p.hasSmoothing then smoothingValue device n profile p memory row block else n.literal 0 := by
  cases smoothing : p.hasSmoothing with
  | «false» =>
    simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,smoothing,Bool.false_eq_true,↓reduceIte,Option.getD]
    rfl
  | «true» =>
    rw [source_smoothing_lowering device p memory row block smoothing,emit_masked_sum device _ positive]
    simp only [smoothing,↓reduceIte,smoothingValue,positive,↓reduceDIte]
    exact smoothing_emit_exact n profile _ _ _ _

theorem source_lse_instruction (device : DeviceFamily) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) :
    (values (algebra device p.logitScale) p memory row block).lse.value=
      Scheduled.lseInstruction (VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits
        (scaled (algebra device p.logitScale) p.logitScale) p.toParameters memory row block) := by
  simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.binary,VeriTile.Bench.Repairs.CrossEntropy1.Source.unary,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.sum,VeriTile.Bench.Repairs.CrossEntropy1.Source.maximum,
    scaled,algebra,Scheduled.algebra,emitSub,emitAdd,exponential_sum,Scheduled.lseInstruction]

theorem source_lse_scalar (device : DeviceFamily) (n : Ports) (profile : Profile) (laws : Peepholes n)
    (p : Parameters) (memory : Memory Compiled) (row block : Nat) :
    let input := fun lane : Fin p.block => decode n profile
      ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
    decode n profile (values (algebra device p.logitScale) p memory row block).lse=
      nativeLSE n profile (nativeScaled n p block input) := by
  unfold decode
  rw [source_lse_instruction,Scheduled.lse_instruction_scalar n profile laws]
  apply congrArg (nativeLSE n profile)
  funext lane
  unfold VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits nativeScaled
  split
  · exact laws.mulComm _ _
  · rfl

theorem loss_z_scalar (device : DeviceFamily) (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (p : Parameters) (memory : Memory Compiled) (row block : Nat) (target : Int)
    (v : VeriTile.Bench.Repairs.CrossEntropy1.Source.ForwardValues Compiled p.block) :
    (decode n profile (lossZ (algebra device p.logitScale) p memory row block target v).1,
      decode n profile (lossZ (algebra device p.logitScale) p memory row block target v).2)=
    nativeLossZ n p target (blockOwns p target block) (decode n profile v.lse)
      (decode n profile (v.sumLogits.getD ((algebra device p.logitScale).literal 0)))
      (decode n profile ((memory p.logits
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)).read .real)) := by
  exact Scheduled.loss_z_scalar n profile laws one .pureTrees p memory row block target v

theorem gradient_scalar (device : DeviceFamily) (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (o : Host.Options) (target : Int) (logit L upstream : Compiled) (col : Nat)
    (commutedFMA commutedOutput : Bool) :
    decode n profile (Host.gradientScalar (algebra device o.logitScale) o target logit L upstream col)=
      nativeGradient n o target (decode n profile logit) (decode n profile L) (decode n profile upstream)
        col commutedFMA commutedOutput := by
  exact Scheduled.gradient_scalar n profile laws one .pureTrees o target logit L upstream col commutedFMA commutedOutput

#axiomsClean emit_masked_sum
#axiomsClean source_smoothing_scalar
#axiomsClean source_lse_scalar
#axiomsClean loss_z_scalar
#axiomsClean gradient_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
