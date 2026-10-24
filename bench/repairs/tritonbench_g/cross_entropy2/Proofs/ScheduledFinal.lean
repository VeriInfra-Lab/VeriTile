import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledGraph
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
open Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations localResult)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

theorem word_host_sum (n : Ports) (profile : Profile) (xs : List Compiled) :
    decode n profile (hostArithmetic.sum xs)=(wordArithmetic n profile).sum (xs.map (decode n profile)) := by
  change reduce profile.hostSumTree n.hostAdd (n.literal 0)
    (List.ofFn (fun i => decode n profile (xs.get i)))=_
  have list : List.ofFn (fun i => decode n profile (xs.get i))=xs.map (decode n profile) := by
    apply List.ext_getElem <;> simp
  rw [list]
  rfl

theorem word_host_lse (n : Ports) (profile : Profile) (xs : List Compiled) :
    decode n profile (hostArithmetic.logsumexp xs)=(wordArithmetic n profile).logsumexp (xs.map (decode n profile)) := by
  change hostLSE n profile (List.ofFn (fun i => decode n profile (xs.get i)))=_
  have list : List.ofFn (fun i => decode n profile (xs.get i))=xs.map (decode n profile) := by
    apply List.ext_getElem <;> simp
  rw [list]
  rfl

theorem word_local_sum (n : Ports) (profile : Profile) (count rows : Nat) (v : Nat → Compiled) (row : Nat) :
    decode n profile (localResult hostArithmetic.sum count rows v row)=
      localResult (wordArithmetic n profile).sum count rows (fun i => decode n profile (v i)) row := by
  unfold localResult
  split
  · rw [word_host_sum]
    simp only [List.map_map,Function.comp_def]
  · rfl

theorem word_local_lse (n : Ports) (profile : Profile) (count rows : Nat) (v : Nat → Compiled) (row : Nat) :
    decode n profile (localResult hostArithmetic.logsumexp count rows v row)=
      localResult (wordArithmetic n profile).logsumexp count rows (fun i => decode n profile (v i)) row := by
  unfold localResult
  split
  · rw [word_host_lse]
    simp only [List.map_map,Function.comp_def]
  · rfl

variable {σ : Type} {obs : Observations Compiled σ} {rt : Runtime Compiled σ} {o : Options}
  {x y : Tensor} {rows cols : Nat} {initial final : World Compiled σ} {returned : Return}
  {e : ForwardData (algebra .aligned65536 o.logitScale) rt o x y rows cols initial returned final}

def nativeOutput (n : Ports) (profile : Profile) (k : ForwardKernelData obs e) (role : Grid.Role) (i : Nat) : Word :=
  alignedRole n profile e.parameters k.ready.memory (i%rows) (i/rows) role

theorem output_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (k : ForwardKernelData obs e) (role : Grid.Role)
    (width : o.width=65536) (smoothing : o.hasSmoothing=Bool.true) :
    (fun i => decode n profile (k.outputValues role i))=nativeOutput n profile k role := by
  funext i
  exact aligned_role_scalar n profile laws one e.parameters k.ready.memory (i%rows) (i/rows) role width smoothing

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
    (k : ForwardKernelData obs e) (width : o.width=65536) (smoothing : o.hasSmoothing=Bool.true) (row : Nat) :
    decode n profile (finalLoss k hostArithmetic row)=nativeFinalLoss n profile k row ∧
    decode n profile (finalLSE k hostArithmetic row)=nativeFinalLSE n profile k row ∧
    decode n profile (finalZ k hostArithmetic row)=nativeFinalZ n profile k row := by
  have losses := congrFun (output_native n profile laws one k .loss width smoothing)
  have lses := congrFun (output_native n profile laws one k .lse width smoothing)
  have zs := congrFun (output_native n profile laws one k .zLoss width smoothing)
  have ls := word_local_sum n profile e.count rows (k.outputValues .loss) row
  have lse := word_local_lse n profile e.count rows (k.outputValues .lse) row
  rw [output_native n profile laws one k .loss width smoothing] at ls
  rw [output_native n profile laws one k .lse width smoothing] at lse
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

#axiomsClean word_host_sum
#axiomsClean word_host_lse
#axiomsClean output_native
#axiomsClean final_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
