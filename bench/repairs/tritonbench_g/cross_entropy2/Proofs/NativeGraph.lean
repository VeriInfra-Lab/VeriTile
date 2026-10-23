import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardScalar
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardScalar

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations localResult)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Complete native block graph: guarded scaled loads, explicit reduction
trees, and the pinned scalar loss/z instructions. -/
def nativeRole (n : Ports) (profile : Profile) (p : Parameters) (memory : Memory Expr)
    (row block : Nat) (role : Grid.Role) : Word :=
  let input := fun lane : Fin p.block => word n profile
    ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
  let xs := nativeScaled n p block input
  let L := nativeLSE n profile xs
  let sums := if p.hasSmoothing then reduce profile.sumTree n.add (n.literal 0) (List.ofFn (fun lane =>
    if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then xs lane else n.literal 0))
    else n.literal 0
  let target := VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row
  let selected := word n profile ((memory p.logits
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)).read .real)
  let pair := nativeLossZ n p target (blockOwns p target block) L sums selected
  match role with
  | .lse => L
  | .loss => pair.1
  | .zLoss => pair.2

theorem role_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (p : Parameters) (memory : Memory Expr) (row block : Nat) (role : Grid.Role) :
    word n profile (Grid.roleValue algebra p memory (row,block) role)=nativeRole n profile p memory row block role := by
  have reductions := forward_reductions_native n profile laws p memory row block
  have pair := loss_z_native n profile laws one p memory row block
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row) (values algebra p memory row block)
  rw [reductions.1,reductions.2] at pair
  cases role with
  | lse => exact reductions.1
  | loss => exact congrArg Prod.fst pair
  | zLoss => exact congrArg Prod.snd pair

/-- Ordinary Torch scalar operations and explicitly selected host trees.
No reduction or final-result equality is assumed. -/
def wordArithmetic (n : Ports) (profile : Profile) : Arithmetic Word where
  zero := n.literal 0
  add := n.hostAdd
  multiply := n.hostMul
  square := fun x => n.hostMul x x
  scale := fun scale x => n.hostMul (n.literal scale) x
  sum := reduce profile.hostSumTree n.hostAdd (n.literal 0)
  logsumexp := hostLSE n profile
  rankSum := reduce profile.hostSumTree n.hostAdd (n.literal 0)

theorem word_barrier (n : Ports) (profile : Profile) (x : Expr) :
    word n profile (.barrier x)=word n profile x := rfl

theorem word_host_sum (n : Ports) (profile : Profile) (xs : List Expr) :
    word n profile (hostArithmetic.sum xs)=(wordArithmetic n profile).sum (xs.map (word n profile)) := by
  change reduce profile.hostSumTree n.hostAdd (n.literal 0)
    (List.ofFn (fun i => word n profile (xs.get i)))=_
  have list : List.ofFn (fun i => word n profile (xs.get i))=xs.map (word n profile) := by
    apply List.ext_getElem <;> simp
  rw [list]
  rfl

theorem word_host_lse (n : Ports) (profile : Profile) (xs : List Expr) :
    word n profile (hostArithmetic.logsumexp xs)=(wordArithmetic n profile).logsumexp (xs.map (word n profile)) := by
  change hostLSE n profile (List.ofFn (fun i => word n profile (xs.get i)))=_
  have list : List.ofFn (fun i => word n profile (xs.get i))=xs.map (word n profile) := by
    apply List.ext_getElem <;> simp
  rw [list]
  rfl

theorem word_local_sum (n : Ports) (profile : Profile) (count rows : Nat) (v : Nat → Expr) (row : Nat) :
    word n profile (localResult hostArithmetic.sum count rows v row)=
      localResult (wordArithmetic n profile).sum count rows (fun i => word n profile (v i)) row := by
  unfold localResult
  split
  · rw [word_host_sum]
    simp only [List.map_map,Function.comp_def]
  · rfl

theorem word_local_lse (n : Ports) (profile : Profile) (count rows : Nat) (v : Nat → Expr) (row : Nat) :
    word n profile (localResult hostArithmetic.logsumexp count rows v row)=
      localResult (wordArithmetic n profile).logsumexp count rows (fun i => word n profile (v i)) row := by
  unfold localResult
  split
  · rw [word_host_lse]
    simp only [List.map_map,Function.comp_def]
  · rfl

variable {σ : Type} {obs : Observations Expr σ} {rt : Runtime Expr σ} {o : Options}
  {x y : Tensor} {rows cols : Nat} {initial final : World Expr σ} {returned : Return}
  {e : ForwardData algebra rt o x y rows cols initial returned final}

def nativeOutput (n : Ports) (profile : Profile) (k : ForwardKernelData obs e) (role : Grid.Role) (i : Nat) : Word :=
  nativeRole n profile e.parameters k.ready.memory (i%rows) (i/rows) role

theorem output_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (k : ForwardKernelData obs e) (role : Grid.Role) :
    (fun i => word n profile (k.outputValues role i))=nativeOutput n profile k role := by
  funext i
  exact role_native n profile laws one e.parameters k.ready.memory (i%rows) (i/rows) role

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
    word n profile (finalLoss k hostArithmetic row)=nativeFinalLoss n profile k row ∧
    word n profile (finalLSE k hostArithmetic row)=nativeFinalLSE n profile k row ∧
    word n profile (finalZ k hostArithmetic row)=nativeFinalZ n profile k row := by
  have losses := congrFun (output_native n profile laws one k .loss)
  have lses := congrFun (output_native n profile laws one k .lse)
  have zs := congrFun (output_native n profile laws one k .zLoss)
  have ls := word_local_sum n profile e.count rows (k.outputValues .loss) row
  have lse := word_local_lse n profile e.count rows (k.outputValues .lse) row
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
      · change n.hostAdd (n.hostAdd (word n profile _) (word n profile _))
          (n.hostMul (n.hostMul (n.literal o.squareScale) (word n profile _)) (word n profile _))=_
        simp only [word_barrier]
        rw [ls,lse]
        rfl
    · unfold aggregateZValue
      split
      · rfl
      · change n.hostMul (n.hostMul (n.literal o.squareScale) (word n profile _)) (word n profile _)=_
        simp only [word_barrier]
        rw [lse]
        rfl

#axiomsClean role_native
#axiomsClean word_host_sum
#axiomsClean word_host_lse
#axiomsClean output_native
#axiomsClean final_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
