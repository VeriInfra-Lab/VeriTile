import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Specialization
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardScalar

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
open Source Host
noncomputable section
set_option maxHeartbeats 10000000

theorem exponential_sum (schedule : DeviceSchedule) (scale : ℝ) {width : Nat}
    (xs : Fin width → Instruction) :
    emitSum schedule scale (fun i => ⟨.unary .exp (xs i),none⟩)=
      ⟨.sum width (fun i => .unary .exp (xs i)),none⟩ := by
  by_cases h : width=65536
  · have pos : 0<width := by omega
    simp [emitSum,atIndex,pos,smoothingInput]
  · simp [emitSum,h]

def lseInstruction {width : Nat} (xs : Fin width → Compiled) : Instruction :=
  .binary .add
    (.unary .log (.sum width (fun i => .unary .exp (.binary .sub (xs i).value
      (.maximum width (fun j => (xs j).value))))))
    (.maximum width (fun j => (xs j).value))

theorem lse_instruction_scalar (n : Ports) (profile : Profile) (laws : Peepholes n)
    {width : Nat} (xs : Fin width → Compiled) :
    runInstruction n profile (lseInstruction xs)=nativeLSE n profile (fun i => decode n profile (xs i)) := by
  exact laws.addComm _ _

theorem source_lse_instruction (schedule : DeviceSchedule) (p : Parameters)
    (memory : Memory Compiled) (row block : Nat) :
    (values (algebra schedule p.logitScale) p memory row block).lse.value=
      lseInstruction (VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits
        (scaled (algebra schedule p.logitScale) p.logitScale) p.toParameters memory row block) := by
  simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.binary,VeriTile.Bench.Repairs.CrossEntropy1.Source.unary,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.sum,VeriTile.Bench.Repairs.CrossEntropy1.Source.maximum,
    scaled,algebra,emitSub,emitAdd,exponential_sum,lseInstruction]

theorem source_lse_scalar (n : Ports) (profile : Profile) (laws : Peepholes n)
    (schedule : DeviceSchedule) (p : Parameters) (memory : Memory Compiled) (row block : Nat) :
    let input := fun lane : Fin p.block => decode n profile
      ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
    decode n profile (values (algebra schedule p.logitScale) p memory row block).lse=
      nativeLSE n profile (nativeScaled n p block input) := by
  unfold decode
  rw [source_lse_instruction,lse_instruction_scalar n profile laws]
  apply congrArg (nativeLSE n profile)
  funext lane
  unfold VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits nativeScaled
  split
  · exact laws.mulComm _ _
  · rfl

def alignedSum (n : Ports) (profile : Profile) (p : Parameters) (memory : Memory Compiled)
    (row block : Nat) : Word :=
  let active := fun lane : Fin p.block =>
    decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols)
  let input := fun lane : Fin p.block =>
    ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value
  alignedSmoothing.run n (n.literal p.logitScale)
    (fun i => runInstruction n profile (boundedInput active input i)) (boundedActive active)

/-- All three block outputs with the aligned smoothing graph inserted into
the scalar loss calculation; LSE retains the pure max and exponential trees. -/
def alignedRole (n : Ports) (profile : Profile) (p : Parameters) (memory : Memory Compiled)
    (row block : Nat) (role : Grid.Role) : Word :=
  let input := fun lane : Fin p.block => decode n profile
    ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
  let L := nativeLSE n profile (nativeScaled n p block input)
  let target := VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row
  let selected := decode n profile ((memory p.logits
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)).read .real)
  let pair := nativeLossZ n p target (blockOwns p target block) L (alignedSum n profile p memory row block) selected
  match role with
  | .lse => L
  | .loss => pair.1
  | .zLoss => pair.2

theorem aligned_role_scalar (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (p : Parameters) (memory : Memory Compiled) (row block : Nat) (role : Grid.Role)
    (width : p.block=65536) (smoothing : p.hasSmoothing=Bool.true) :
    decode n profile (Grid.roleValue (algebra .aligned65536 p.logitScale) p memory (row,block) role)=
      alignedRole n profile p memory row block role := by
  have lse := source_lse_scalar n profile laws .aligned65536 p memory row block
  have sum := source_smoothing_scalar n profile p memory row block width smoothing
  have pair := loss_z_scalar n profile laws one .aligned65536 p memory row block
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row)
    (values (algebra .aligned65536 p.logitScale) p memory row block)
  rw [lse,sum] at pair
  cases role with
  | lse => exact lse
  | loss => exact congrArg Prod.fst pair
  | zLoss => exact congrArg Prod.snd pair

#axiomsClean exponential_sum
#axiomsClean source_lse_scalar
#axiomsClean aligned_role_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
