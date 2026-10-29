import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Profiled
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledFinal

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
open Source Host
open Scheduled (decode)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations localResult)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

def nativeRole (device : DeviceFamily) (n : Ports) (profile : Profile) (p : Parameters) (memory : Memory Compiled)
    (row block : Nat) (role : Grid.Role) : Word :=
  let input := fun lane : Fin p.block => decode n profile
    ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
  let L := nativeLSE n profile (nativeScaled n p block input)
  let target := VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row
  let selected := decode n profile ((memory p.logits
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)).read .real)
  let pair := nativeLossZ n p target (blockOwns p target block) L (if p.hasSmoothing then smoothingValue device n profile p memory row block else n.literal 0) selected
  match role with
  | .lse => L
  | .loss => pair.1
  | .zLoss => pair.2

theorem role_scalar (device : DeviceFamily) (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (p : Parameters) (memory : Memory Compiled) (row block : Nat) (role : Grid.Role)
    (positive : 0<p.block) :
    decode n profile (Grid.roleValue (algebra device p.logitScale) p memory (row,block) role)=
      nativeRole device n profile p memory row block role := by
  have lse := source_lse_scalar device n profile laws p memory row block
  have sum := source_smoothing_scalar device n profile p memory row block positive
  have pair := loss_z_scalar device n profile laws one p memory row block
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row)
    (values (algebra device p.logitScale) p memory row block)
  rw [lse,sum] at pair
  cases role with
  | lse => exact lse
  | loss => exact congrArg Prod.fst pair
  | zLoss => exact congrArg Prod.snd pair

variable {device : DeviceFamily} {σ : Type} {obs : Observations Compiled σ} {rt : Runtime Compiled σ} {o : Options}
  {x y : Tensor} {rows cols : Nat} {initial final : World Compiled σ} {returned : Return}
  {e : ForwardData (algebra device o.logitScale) rt o x y rows cols initial returned final}

def nativeOutput (n : Ports) (profile : Profile) (k : ForwardKernelData obs e) (role : Grid.Role) (i : Nat) : Word :=
  nativeRole device n profile e.parameters k.ready.memory (i%rows) (i/rows) role

theorem output_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (k : ForwardKernelData obs e) (role : Grid.Role) :
    (fun i => decode n profile (k.outputValues role i))=nativeOutput n profile k role := by
  funext i
  exact role_scalar device n profile laws one e.parameters k.ready.memory (i%rows) (i/rows) role e.admitted.2.2.2.2.2.1

def nativeFinalLoss (n : Ports) (profile : Profile) (k : ForwardKernelData obs e) : Nat → Word :=
  if e.split then aggregateLossValue (wordArithmetic n profile) o
    (localResult (wordArithmetic n profile).sum e.count rows (nativeOutput n profile k .loss))
    (localResult (wordArithmetic n profile).logsumexp e.count rows (nativeOutput n profile k .lse))
    (obs.integer initial y)
  else nativeOutput n profile k .loss
def nativeFinalLSE (n : Ports) (profile : Profile) (k : ForwardKernelData obs e) : Nat → Word :=
  if e.split then localResult (wordArithmetic n profile).logsumexp e.count rows (nativeOutput n profile k .lse)
  else nativeOutput n profile k .lse
def nativeFinalZ (n : Ports) (profile : Profile) (k : ForwardKernelData obs e) : Nat → Word :=
  if e.split then aggregateZValue (wordArithmetic n profile) o
    (localResult (wordArithmetic n profile).logsumexp e.count rows (nativeOutput n profile k .lse))
    (obs.integer initial y)
  else nativeOutput n profile k .zLoss

/-- Every complete forward role follows through the actual host sum,
stable logsumexp, separate products, additions and ignored-row masks. -/
theorem final_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (k : ForwardKernelData obs e) (row : Nat) :
    decode n profile (finalLoss k Scheduled.hostArithmetic row)=nativeFinalLoss n profile k row ∧
    decode n profile (finalLSE k Scheduled.hostArithmetic row)=nativeFinalLSE n profile k row ∧
    decode n profile (finalZ k Scheduled.hostArithmetic row)=nativeFinalZ n profile k row := by
  have losses := congrFun (output_native n profile laws one k .loss)
  have lses := congrFun (output_native n profile laws one k .lse)
  have zs := congrFun (output_native n profile laws one k .zLoss)
  have ls := Scheduled.word_local_sum n profile e.count rows (k.outputValues .loss) row
  have lse := Scheduled.word_local_lse n profile e.count rows (k.outputValues .lse) row
  rw [output_native n profile laws one k .loss] at ls
  rw [output_native n profile laws one k .lse] at lse
  cases split : e.split with
  | «false» => exact ⟨by simpa [finalLoss,nativeFinalLoss,split] using losses row,
      by simpa [finalLSE,nativeFinalLSE,split] using lses row,
      by simpa [finalZ,nativeFinalZ,split] using zs row⟩
  | «true» =>
    simp only [finalLoss,finalLSE,finalZ,nativeFinalLoss,nativeFinalLSE,nativeFinalZ,split,↓reduceIte]
    refine ⟨?_,lse,?_⟩
    · unfold aggregateLossValue
      split
      · rfl
      · change n.hostAdd (n.hostAdd (decode n profile _) (decode n profile _))
          (n.hostMul (n.hostMul (n.literal o.squareScale) (decode n profile _)) (decode n profile _))=_
        rw [ls,lse]
        rfl
    · unfold aggregateZValue
      split
      · rfl
      · change n.hostMul (n.hostMul (n.literal o.squareScale) (decode n profile _)) (decode n profile _)=_
        rw [lse]
        rfl

#axiomsClean role_scalar
#axiomsClean output_native
#axiomsClean final_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
