import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.WholeHost
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Domain
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Host
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000

/-- Decode the six supplied disjoint allocations. Input pointers denote their
own region origin; the output pointer additionally retains its storage offset.
Metadata uses ordinary integer decoding, never an output or reduction oracle. -/
def decodeMemory {α} (index : α → Nat) (m : Memory α) : Input α where
  logics := m "Logics"
  values := m "V"
  starts := fun a => index (m "Starts" a)
  counts := fun a => index (m "Lengths" a)
  pages := fun a => index (m "Locations" a)

/-- Exact-real metadata representation used only by the mathematical theorem.
The native entry instead decodes the actual 32-bit integer storage word. -/
def realIndex (x : WithBot ℝ) : Nat := Int.toNat ⌊x.unbotD 0⌋

/-- Writing through the supplied Out pointer preserves every decoded input,
including all starts, lengths and page indices; no input snapshot is inserted. -/
theorem decode_putOutput {α} (index : α → Nat) (base : Nat) (m : Memory α) (out : Nat → α) :
    decodeMemory index (putOutput base m out)=decodeMemory index m := by
  simp only [decodeMemory]
  congr 1

/-- The actual typed source launch consumes these decoded allocation cells.
Signed-i32 nonnegative metadata is represented by the source's Nat/Int tags;
Admissible establishes the active and padded fixed-width arithmetic bounds. -/
def memoryLaunch {α} [Inhabited α] (index : α → Nat) (c : Call) (base : Nat)
    (m : Memory α) (j : Job) : State α :=
  launch c (decodeMemory index m) (readOutput base m) j

theorem launch_binding {α} [Inhabited α] (index : α → Nat) (c : Call) (base : Nat)
    (m : Memory α) (j : Job) (a : Nat) :
    (memoryLaunch index c base m j).mem "Logics" a=.mk .fp32 (m "Logics" a) ∧
    (memoryLaunch index c base m j).mem "V" a=.mk .fp32 (m "V" a) ∧
    (memoryLaunch index c base m j).mem "Out" a=.mk .fp32 (m "Out" (base+a)) ∧
    (memoryLaunch index c base m j).mem "Starts" a=.mk .nat (index (m "Starts" a)) ∧
    (memoryLaunch index c base m j).mem "Lengths" a=.mk .nat (index (m "Lengths" a)) ∧
    (memoryLaunch index c base m j).mem "Locations" a=.mk .int (Int.ofNat (index (m "Locations" a))) := by
  simp [memoryLaunch,launch,decodeMemory,readOutput]

/-- Allocation interpretation of the complete typed source result. The source
itself proves every non-output cell unchanged; putOutput only translates the
output pointer offset back to its caller allocation, retaining all other bytes.
This bridge supplies no source result formula or numerical callback. -/
theorem execution_bridge {α} [Inhabited α] (M : Algebra α) (index : α → Nat)
    (c : Call) (base : Nat) (m : Memory α) (j : Job) (legal : Legal c j) (u : State α)
    (run : FP.Structural.exec M (FPValue.source (parameters c)) (memoryLaunch index c base m j)=some u) :
    (∀ (r : RegionName) a, r≠"Out" → u.mem r a=(memoryLaunch index c base m j).mem r a) ∧
    (∀ r a, Outside c base r a →
      putOutput base m (fun q => (u.mem "Out" q).read .fp32) r a=m r a) := by
  obtain ⟨v,hv,post⟩ := FPValue.correctness M (parameters c) (memoryLaunch index c base m j)
    (by change 0 < 64;decide) (source_injective c (decodeMemory index m) (readOutput base m) j)
  have eq : v=u := Option.some.inj (hv.symm.trans run)
  subst v
  constructor
  · intro r a hr
    exact post.2 r a (Or.inl hr)
  · intro r a outside
    by_cases h : r="Out" ∧ base ≤ a
    · have off : Off c (workset c) (a-base) := by
        rcases outside with hr | before | off
        · exact (hr h.1).elim
        · omega
        · exact off
      have saved := post.2 "Out" (a-base) (Or.inr (by
        intro i eq
        apply off j ((mem_workset c j).mpr legal)
        exact ⟨i.1.val,i.1.isLt,eq.trans (source_address c (decodeMemory index m) (readOutput base m) j i)⟩))
      have value := congrArg (Cell.read .fp32) saved
      simpa [putOutput,h,h.1,memoryLaunch,launch,readOutput,Nat.add_sub_of_le h.2] using value
    · simp [putOutput,h]

/-- Local source execution in the caller's allocation interpretation. -/
def memoryEvaluate {α} [Inhabited α] (M : Algebra α) (index : α → Nat)
    (c : Call) (base : Nat) (j : Job) (m : Memory α) : Option (Nat → α) :=
  (FP.Structural.exec M (FPValue.source (parameters c)) (memoryLaunch index c base m j)).map
    (fun u a => (u.mem "Out" a).read .fp32)

theorem memory_evaluate {α} [Inhabited α] (M : Algebra α) (index : α → Nat)
    (c : Call) (base : Nat) (j : Job) (m : Memory α) (out : Nat → α) :
    memoryEvaluate M index c base j (putOutput base m out)=evaluate M c (decodeMemory index m) j out := by
  simp only [memoryEvaluate,memoryLaunch,decode_putOutput,put_read,evaluate]

/-- The runtime succeeds by executing the complete typed source with inputs
read from the current caller allocations; local errors can partially write
only that job's output. This contains no expected output or numerical law. -/
structure AllocationRuntime {α} (c : Call) (base : Nat) (initial : Memory α)
    (fullEvaluate : Job → Memory α → Option (Nat → α))
    (invoke : Job → (Nat → α) → SoftmaxTriton1.Result (Nat → α) Unit) : Prop where
  success : ∀ j s t, Legal c j → invoke j s=.ok () t → fullEvaluate j (putOutput base initial s)=some t
  error : ∀ j s e t, Legal c j → invoke j s=.error e t → Frame c [j] s t

theorem allocation_local {α} [Inhabited α] (M : Algebra α) (index : α → Nat)
    (c : Call) (base : Nat) (initial : Memory α)
    (invoke : Job → (Nat → α) → SoftmaxTriton1.Result (Nat → α) Unit)
    (rt : AllocationRuntime c base initial (memoryEvaluate M index c base) invoke) :
    LocalRuntime M c (decodeMemory index initial) invoke := by
  refine ⟨?_,rt.error⟩
  intro j s t hj run
  simpa only [memory_evaluate] using rt.success j s t hj run

/-- Real metadata really encodes nonnegative integral ABI words at every read;
masked page cells are unconstrained, as in the original caller interface. -/
structure RealMetadataEncoding (c : Call) (m : Memory (WithBot ℝ)) : Prop where
  starts : ∀ b, b < c.batch → m "Starts" b=↑((realIndex (m "Starts" b)):ℝ)
  counts : ∀ b, b < c.batch → m "Lengths" b=↑((realIndex (m "Lengths" b)):ℝ)
  pages : ∀ j, Legal c j → ∀ k, k < (decodeMemory realIndex m).counts j.batch →
    let a := pageOffset c (decodeMemory realIndex m) j k
    m "Locations" a=↑((realIndex (m "Locations" a)):ℝ)

#axiomsClean memory_evaluate
#axiomsClean allocation_local
#axiomsClean decode_putOutput
#axiomsClean launch_binding
#axiomsClean execution_bridge
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Host
