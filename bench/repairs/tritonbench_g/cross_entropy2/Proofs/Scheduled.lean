import bench.repairs.tritonbench_g.cross_entropy2.Proofs.FusedReduction
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardScalar

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
open Source Reduction
noncomputable section
set_option maxHeartbeats 10000000

instance : Inhabited Compiled := ⟨⟨.literal 0,none⟩⟩

inductive DeviceSchedule where
  | pureTrees
  | aligned65536
  deriving DecidableEq

def decode (n : Ports) (profile : Profile) (x : Compiled) : Word :=
  runInstruction n profile x.value

def productInput (x : Compiled) : Instruction :=
  match x.product with
  | some (a,_) => a
  | none => .literal 0

/-- Source sum operands are either exponential results, scaled loads, or
literal padding. This tag selects only the smoothing sum. The live mask is
separately recovered from the product tag, which a source load cannot forge. -/
def smoothingInput (x : Compiled) : Bool :=
  match x.product with
  | some _ => Bool.true
  | none => match x.value with
    | .literal _ => Bool.true
    | _ => Bool.false

def atIndex {width : Nat} (xs : Fin width → Compiled) (i : Nat) : Compiled :=
  if h : i<width then xs ⟨i,h⟩ else ⟨.literal 0,none⟩

def emitSum (schedule : DeviceSchedule) (scale : ℝ) {width : Nat}
    (xs : Fin width → Compiled) : Compiled :=
  if schedule=.aligned65536 ∧ width=65536 ∧ smoothingInput (atIndex xs 0)=Bool.true then
    ⟨alignedSmoothing.emit (.literal scale) (fun i => productInput (atIndex xs i))
      (fun i => (atIndex xs i).product.isSome),none⟩
  else ⟨.sum width (fun i => (xs i).value),none⟩

/-- Source-level arithmetic with an explicit compiler schedule. Products
retain operands until their consuming instruction; reduction lowering may
consume those operands as FMA arguments. The old pure Expr model is retained. -/
def algebra (schedule : DeviceSchedule) (scale : ℝ) : FP32 Compiled where
  literal := fun x => ⟨.literal x,none⟩
  fromNat := fun x => ⟨.natural x,none⟩
  negInf := ⟨.negativeInfinity,none⟩
  load := fun x => ⟨x.value,none⟩
  add := emitAdd
  sub := emitSub
  mul := fun x y => ⟨.binary .mul x.value y.value,some (x.value,y.value)⟩
  div := fun x y => ⟨.binary .div x.value y.value,none⟩
  exp := fun x => ⟨.unary .exp x.value,none⟩
  log := fun x => ⟨.unary .log x.value,none⟩
  sum := emitSum schedule scale
  maximum := fun {width} xs => ⟨.maximum width (fun i => (xs i).value),none⟩

def hostArithmetic : Host.Arithmetic Compiled where
  zero := ⟨.literal 0,none⟩
  add := fun x y => ⟨.hostAdd x.value y.value,none⟩
  multiply := fun x y => ⟨.hostMultiply x.value y.value,none⟩
  square := fun x => ⟨.hostSquare x.value,none⟩
  scale := fun scale x => ⟨.hostScale scale x.value,none⟩
  sum := fun xs => ⟨.hostSum xs.length (fun i => (xs.get i).value),none⟩
  logsumexp := fun xs => ⟨.hostLSE xs.length (fun i => (xs.get i).value),none⟩
  rankSum := fun xs => ⟨.rankSum xs.length (fun i => (xs.get i).value),none⟩

def maskedProduct (scale : ℝ) (active : Bool) (input : Instruction) : Compiled :=
  if active then ⟨.binary .mul input (.literal scale),some (input,.literal scale)⟩
  else ⟨.literal 0,none⟩

theorem masked_product_input (scale : ℝ) (active : Bool) (input : Instruction) :
    productInput (maskedProduct scale active input)=(if active then input else .literal 0) := by
  cases active <;> rfl
theorem masked_product_active (scale : ℝ) (active : Bool) (input : Instruction) :
    (maskedProduct scale active input).product.isSome=active := by
  cases active <;> rfl
theorem masked_product_tag (scale : ℝ) (active : Bool) (input : Instruction) :
    smoothingInput (maskedProduct scale active input)=Bool.true := by
  cases active <;> rfl

def boundedActive {width : Nat} (active : Fin width → Bool) (i : Nat) : Bool :=
  if h : i<width then active ⟨i,h⟩ else Bool.false
def boundedInput {width : Nat} (active : Fin width → Bool) (input : Fin width → Instruction) (i : Nat) : Instruction :=
  if h : i<width then if active ⟨i,h⟩ then input ⟨i,h⟩ else .literal 0 else .literal 0

/-- The source masks are recovered exactly, including inactive groups. The
unselected raw input is zero, as in the pinned masked PTX loads. -/
theorem emit_masked_sum (scale : ℝ) (active : Fin 65536 → Bool) (input : Fin 65536 → Instruction) :
    emitSum .aligned65536 scale (fun i => maskedProduct scale (active i) (input i))=
      ⟨alignedSmoothing.emit (.literal scale) (boundedInput active input) (boundedActive active),none⟩ := by
  have tag : smoothingInput (atIndex (fun i => maskedProduct scale (active i) (input i)) 0)=Bool.true := by
    simp only [atIndex,show 0<65536 by decide,↓reduceDIte,masked_product_tag]
  simp only [emitSum,tag,and_self,ite_true]
  congr 2
  · funext i
    unfold atIndex boundedInput
    split
    · exact masked_product_input _ _ _
    · rfl
  · funext i
    unfold atIndex boundedActive
    split
    · exact masked_product_active _ _ _
    · rfl

theorem masked_sum_scalar (n : Ports) (profile : Profile) (scale : ℝ)
    (active : Fin 65536 → Bool) (input : Fin 65536 → Instruction) :
    decode n profile (emitSum .aligned65536 scale (fun i => maskedProduct scale (active i) (input i)))=
      alignedSmoothing.run n (n.literal scale)
        (fun i => runInstruction n profile (boundedInput active input i)) (boundedActive active) := by
  rw [emit_masked_sum]
  exact smoothing_emit_exact n profile _ _ _ _

/-- This is the actual smoothing operand vector generated by Source.values,
not an independently chosen reduction output. -/
theorem source_smoothing_operands (schedule : DeviceSchedule) (p : Parameters) (memory : Memory Compiled)
    (row block : Nat) (lane : Fin p.block) :
    (if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then
      VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled (algebra schedule p.logitScale) p.logitScale)
        p.toParameters memory row block lane else (algebra schedule p.logitScale).literal 0)=
    maskedProduct p.logitScale
      (decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols))
      ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value := by
  by_cases h : VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols
  all_goals simp [h,maskedProduct,VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits,scaled,algebra,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.load]

/-- Complete source smoothing reduction now decodes to the fused graph. -/
theorem source_smoothing_lowering (n : Ports) (profile : Profile) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) (width : p.block=65536) (smoothing : p.hasSmoothing=Bool.true) :
    decode n profile ((values (algebra .aligned65536 p.logitScale) p memory row block).sumLogits.getD
      ((algebra .aligned65536 p.logitScale).literal 0))=
    decode n profile (emitSum .aligned65536 p.logitScale (fun lane : Fin p.block =>
      maskedProduct p.logitScale
        (decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols))
        ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
          (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value)) := by
  have operands := funext (source_smoothing_operands .aligned65536 p memory row block)
  simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,smoothing,↓reduceIte,Option.getD]
  change decode n profile (emitSum .aligned65536 p.logitScale _)=_
  exact congrArg (fun xs => decode n profile (emitSum .aligned65536 p.logitScale xs)) operands

theorem masked_sum_width_scalar (n : Ports) (profile : Profile) (scale : ℝ) {width : Nat}
    (h : width=65536) (active : Fin width → Bool) (input : Fin width → Instruction) :
    decode n profile (emitSum .aligned65536 scale (fun i => maskedProduct scale (active i) (input i)))=
      alignedSmoothing.run n (n.literal scale)
        (fun i => runInstruction n profile (boundedInput active input i)) (boundedActive active) := by
  subst width
  exact masked_sum_scalar n profile scale active input

/-- The complete smoothing expression has the explicit three-FMA and masked
tail scalar interpretation, with the actual source addresses and guards. -/
theorem source_smoothing_scalar (n : Ports) (profile : Profile) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) (width : p.block=65536) (smoothing : p.hasSmoothing=Bool.true) :
    let active := fun lane : Fin p.block =>
      decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols)
    let input := fun lane : Fin p.block =>
      ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value
    decode n profile ((values (algebra .aligned65536 p.logitScale) p memory row block).sumLogits.getD
      ((algebra .aligned65536 p.logitScale).literal 0))=
      alignedSmoothing.run n (n.literal p.logitScale)
        (fun i => runInstruction n profile (boundedInput active input i)) (boundedActive active) := by
  exact (source_smoothing_lowering n profile p memory row block width smoothing).trans
    (masked_sum_width_scalar n profile p.logitScale width _ _)

theorem loss_z_scalar (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (schedule : DeviceSchedule) (p : Parameters) (memory : Memory Compiled) (row block : Nat) (target : Int)
    (v : VeriTile.Bench.Repairs.CrossEntropy1.Source.ForwardValues Compiled p.block) :
    (decode n profile (lossZ (algebra schedule p.logitScale) p memory row block target v).1,
      decode n profile (lossZ (algebra schedule p.logitScale) p memory row block target v).2)=
    nativeLossZ n p target (blockOwns p target block) (decode n profile v.lse)
      (decode n profile (v.sumLogits.getD ((algebra schedule p.logitScale).literal 0)))
      (decode n profile ((memory p.logits
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)).read .real)) := by
  change n.natural 1=n.literal 1 at one
  by_cases ignored : target=p.ignored
  · simp [lossZ,nativeLossZ,ignored,decode,fp,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,algebra,runInstruction]
  · cases split : p.split <;> cases smoothing : p.hasSmoothing <;> by_cases owner : blockOwns p target block
    all_goals simp only [lossZ,nativeLossZ,ignored,owner,split,smoothing,↓reduceIte,
      Bool.false_eq_true,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,
      VeriTile.Bench.Repairs.CrossEntropy1.Source.nat,VeriTile.Bench.Repairs.CrossEntropy1.Source.load,
      VeriTile.Bench.Repairs.CrossEntropy1.Source.binary,algebra,decode,emitAdd,emitSub,runInstruction]
    all_goals try rw [laws.negFmaExchange _ (n.literal p.logitScale)]
    all_goals simp only [one,laws.fmaComm (n.mul _ _) _,laws.mulComm (n.mul _ _) _,laws.mulComm _ (n.literal p.logitScale)]

theorem gradient_scalar (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (schedule : DeviceSchedule) (o : Host.Options) (target : Int) (logit L upstream : Compiled) (col : Nat)
    (commutedFMA commutedOutput : Bool) :
    decode n profile (Host.gradientScalar (algebra schedule o.logitScale) o target logit L upstream col)=
      nativeGradient n o target (decode n profile logit) (decode n profile L) (decode n profile upstream)
        col commutedFMA commutedOutput := by
  change n.natural 1=n.literal 1 at one
  have logitComm := laws.mulComm (decode n profile logit) (n.literal o.logitScale)
  have upstreamComm := laws.mulComm (decode n profile upstream) (n.literal o.logitScale)
  by_cases ignored : target=o.ignored
  · simp only [Host.gradientScalar,nativeGradient,ignored,↓reduceIte,decode,algebra,runInstruction]
  · cases smoothing : o.hasSmoothing <;> by_cases hit : (col : Int)=target-o.start <;>
      cases commutedFMA <;> cases commutedOutput
    all_goals simp only [Host.gradientScalar,nativeGradient,ignored,smoothing,hit,↓reduceIte,
      Bool.false_eq_true,decode,algebra,emitAdd,emitSub,runInstruction]
    all_goals simp only [one,laws.twice,laws.minusOne,decode] at logitComm upstreamComm ⊢
    all_goals rw [logitComm,upstreamComm]
    all_goals try rw [laws.fmaComm (n.mul (n.add _ _) _) _]
    all_goals first | rfl | exact laws.mulComm _ _

#axiomsClean emit_masked_sum
#axiomsClean masked_sum_scalar
#axiomsClean source_smoothing_operands
#axiomsClean source_smoothing_scalar
#axiomsClean loss_z_scalar
#axiomsClean gradient_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
