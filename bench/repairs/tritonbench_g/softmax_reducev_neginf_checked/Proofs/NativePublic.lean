import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Completion
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.ABI
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked
open VeriTile.Triton FP.Structural
open Host
noncomputable section
set_option maxHeartbeats 1000000

/-- Concrete scalar-graph source law, derived from the full typed evaluator. -/
def nativeSourceLaw (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (x : Input NativeGraph.Word) :
    SourceLawGeneral c
      (fun j => NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops c x j)
      (fun j => NativeGraph.row (NativeGraph.profilePlan (profiles j)) ops c x j) :=
  ⟨fun j s t run => NativeGraph.source_law (NativeGraph.profilePlan (profiles j)) ops c x j s t run⟩

/-- The exact native instruction graph at every logical output, complete
(batch,head) host launch, supplied-output offset, Python None and full frame. -/
theorem native_return_of_input (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (x : Input NativeGraph.Word) (base : Nat)
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : RuntimeGeneral c (fun j => NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops c x j) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial final : Memory NativeGraph.Word)
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions →
      final "Out" (base+address c j d)=NativeGraph.row (NativeGraph.profilePlan (profiles j)) ops c x j d) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) :=
  whole_general c _ _ (nativeSourceLaw profiles ops c x) base invoke rt schedule complete initial final run

/-- Seven reached scalar instruction families are local model boundaries. Every
reduction, FMA selection, physical replica, loop and final store is already
constructed and executed by the public source/native connection. -/
theorem fpequiv_of_input (profiles : Job → NativeGraph.Profile)
    (native software : NativeGraph.Instructions) (scalar : NativeGraph.InstructionAgreement native software)
    (c : Call) (x : Input NativeGraph.Word) (base : Nat)
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : RuntimeGeneral c (fun j => NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) native c x j) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial final : Memory NativeGraph.Word)
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions →
      final "Out" (base+address c j d)=NativeGraph.row (NativeGraph.profilePlan (profiles j)) software c x j d) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) := by
  have result := native_return_of_input profiles native c x base invoke rt schedule complete initial final run
  refine ⟨fun j hj d hd => ?_,result.2⟩
  exact (result.1 j hj d hd).trans (NativeGraph.row_agreement (NativeGraph.profilePlan (profiles j)) native software scalar c x j d)

/-- Exact byte-frame on every host outcome, including partial launch errors. -/
theorem all_outcomes_frame_of_input (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (x : Input NativeGraph.Word) (base : Nat)
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : RuntimeGeneral c (fun j => NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops c x j) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial : Memory NativeGraph.Word)
    (r : RegionName) (a : Nat) (outside : Outside c base r a) :
    (wrapper c base schedule invoke initial).state r a=initial r a :=
  all_outcomes_general c _ _ (nativeSourceLaw profiles ops c x) base invoke rt schedule complete initial r a outside

/-- Actual nonnegative signed-i32 metadata words and FP32 input words from the
same caller memory. Admissible bounds exclude negative/overflowed active indices. -/
def nativeInput (m : Memory NativeGraph.Word) : Input NativeGraph.Word :=
  decodeMemory BitVec.toNat m

/-- Native staging from the actual FP32/integer caller storage. Each initial
float word becomes an atom; metadata is decoded from that same word. -/
def nativeMemoryEvaluate (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (base : Nat) (j : Job) (m : Memory NativeGraph.Word) : Option (Nat → NativeGraph.Word) :=
  (FP.Structural.exec (NativeGraph.algebra (NativeGraph.profilePlan (profiles j))) (FPValue.source (parameters c))
    (memoryLaunch (fun node => (NativeGraph.eval ops (node 0)).toNat) c base
      (fun r a => NativeGraph.atom (m r a)) j)).map
    (NativeGraph.observe (NativeGraph.profilePlan (profiles j)) ops c j)

theorem native_memory_evaluate (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (base : Nat) (j : Job) (m : Memory NativeGraph.Word) (out : Nat → NativeGraph.Word) :
    nativeMemoryEvaluate profiles ops c base j (putOutput base m out)=
      NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops c (nativeInput m) j out := by
  have staged : (fun r a => NativeGraph.atom (putOutput base m out r a))=
      putOutput base (fun r a => NativeGraph.atom (m r a)) (fun a => NativeGraph.atom (out a)) := by
    funext r a
    by_cases h : r="Out" ∧ base ≤ a <;> simp [putOutput,h]
  simp only [nativeMemoryEvaluate,staged,memoryLaunch,decode_putOutput,put_read]
  rfl

theorem native_allocation_local (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (base : Nat) (initial : Memory NativeGraph.Word)
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : AllocationRuntime c base initial (nativeMemoryEvaluate profiles ops c base) invoke) :
    RuntimeGeneral c (fun j => NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops c (nativeInput initial) j) invoke := by
  refine ⟨?_,rt.error⟩
  intro j s t hj run
  simpa only [native_memory_evaluate] using rt.success j s t hj run

/-- Complete caller allocation result of the actual source/native graph. The
per-job profile and runtime link are structural; they contain no output oracle. -/
theorem native_return (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (base : Nat) (initial final : Memory NativeGraph.Word)
    (_valid : Admissible c (nativeInput initial))
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : AllocationRuntime c base initial (nativeMemoryEvaluate profiles ops c base) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c))
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions →
      final "Out" (base+address c j d)=NativeGraph.row (NativeGraph.profilePlan (profiles j)) ops c (nativeInput initial) j d) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) :=
  native_return_of_input profiles ops c (nativeInput initial) base invoke
    (native_allocation_local profiles ops c base initial invoke rt) schedule complete initial final run

/-- Seven local instruction-conformance fields, with all input/metadata reads
bound to the initial caller allocations and all output channels/frame proved. -/
theorem fpequiv (profiles : Job → NativeGraph.Profile)
    (native software : NativeGraph.Instructions) (scalar : NativeGraph.InstructionAgreement native software)
    (c : Call) (base : Nat) (initial final : Memory NativeGraph.Word)
    (_valid : Admissible c (nativeInput initial))
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : AllocationRuntime c base initial (nativeMemoryEvaluate profiles native c base) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c))
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions →
      final "Out" (base+address c j d)=NativeGraph.row (NativeGraph.profilePlan (profiles j)) software c (nativeInput initial) j d) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) :=
  fpequiv_of_input profiles native software scalar c (nativeInput initial) base invoke
    (native_allocation_local profiles native c base initial invoke rt) schedule complete initial final run

/-- Allocation frame for every successful or partial-error outcome. -/
theorem all_outcomes_frame (profiles : Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Call) (base : Nat) (initial : Memory NativeGraph.Word)
    (_valid : Admissible c (nativeInput initial))
    (invoke : Job → (Nat → NativeGraph.Word) → SoftmaxTriton1.Result (Nat → NativeGraph.Word) Unit)
    (rt : AllocationRuntime c base initial (nativeMemoryEvaluate profiles ops c base) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c))
    (r : RegionName) (a : Nat) (outside : Outside c base r a) :
    (wrapper c base schedule invoke initial).state r a=initial r a :=
  all_outcomes_frame_of_input profiles ops c (nativeInput initial) base invoke
    (native_allocation_local profiles ops c base initial invoke rt) schedule complete initial r a outside

#axiomsClean native_memory_evaluate
#axiomsClean native_allocation_local
#axiomsClean native_return
#axiomsClean fpequiv
#axiomsClean all_outcomes_frame
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked
