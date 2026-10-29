import bench.tritonbench_g.rmsnorm_implementation.Proofs.WholeMath
import bench.tritonbench_g.rmsnorm_implementation.Proofs.WholeIndex

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

/-- Fixed observed graph profile. Specialization-specific complete reduction
order is supplied as a tree, never as a reduction-result callback. -/
def observedGraphProfile : String :=
  "unchanged rmsnorm_wrapper/rmsnorm_triton; FP32; Triton 3.7.1; cuda sm90; BLOCK_N_SIZE=4096; num_warps=16; num_stages=3; enable_fp_fusion=True; signed-i32 dynamic stride ABI; specialization-specific complete scalar-add reduction tree"

def defaultEpsilon : ℝ := 4722366482869645/4722366482869645213696

structure Runtime (α : Type) (p : Parameters) (entry : Memory α) (preexisting : Nat → Prop) where
  allocate : Except String (Allocated α p entry preexisting)
  setup : (a : Allocated α p entry preexisting) → Except String (Schedule p)

inductive Outcome (α : Type) where
  | shapeFailure
  | allocationFailure (reason : String)
  | weightRankFailure (out : Tensor) (memory : Memory α)
  | runtimeFailure (reason : String) (out : Tensor) (memory : Memory α)
  | kernelFailure
  | returned (out : Tensor) (memory : Memory α)

/-- Full Python branch order. The allocator and launch setup have no access to
a final result. Every successful result comes from both source loops. -/
def executeHost (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry preexisting) : Outcome α :=
  if p.input.shape.length = 3 then
    match rt.allocate with
    | .error reason => .allocationFailure reason
    | .ok a =>
      if p.weights.shape.length = 1 then
        match rt.setup a with
        | .error reason => .runtimeFailure reason p.output a.memory
        | .ok schedule =>
          match entryRun Q red p a u schedule with
          | none => .kernelFailure
          | some final => .returned p.output final
      else .weightRankFailure p.output a.memory
  else .shapeFailure

def ReturnedSpec (p : Parameters) (formula : Nat → Nat → Nat → α)
    (a : Allocated α p entry preexisting) (final : Memory α) : Prop :=
  p.output.shape = p.input.shape ∧ p.output.dtype = p.input.dtype ∧
  p.output.device = p.input.device ∧
  (∀ b m k, b < p.batches → m < p.rows → k < p.width →
    final p.output.allocation (outputOffset p b m k) = formula b m k) ∧
  (∀ allocation offset, Outside p allocation offset → final allocation offset = a.memory allocation offset) ∧
  (∀ allocation offset, preexisting allocation → final allocation offset = entry allocation offset)

def HostSpec (p : Parameters) (formula : Nat → Nat → Nat → α)
    (rt : Runtime α p entry preexisting) (result : Outcome α) : Prop :=
  (p.input.shape.length ≠ 3 ∧ result = .shapeFailure) ∨
  (∃ reason, p.input.shape.length = 3 ∧ rt.allocate = .error reason ∧ result = .allocationFailure reason) ∨
  (∃ a, p.input.shape.length = 3 ∧ rt.allocate = .ok a ∧ p.weights.shape.length ≠ 1 ∧
    result = .weightRankFailure p.output a.memory ∧
    ∀ allocation offset, preexisting allocation → a.memory allocation offset = entry allocation offset) ∨
  (∃ a reason, p.input.shape.length = 3 ∧ rt.allocate = .ok a ∧ p.weights.shape.length = 1 ∧
    rt.setup a = .error reason ∧ result = .runtimeFailure reason p.output a.memory ∧
    ∀ allocation offset, preexisting allocation → a.memory allocation offset = entry allocation offset) ∨
  (∃ a schedule final, p.input.shape.length = 3 ∧ rt.allocate = .ok a ∧ p.weights.shape.length = 1 ∧
    rt.setup a = .ok schedule ∧ result = .returned p.output final ∧ ReturnedSpec p formula a final)

def logicalTensor (Q : Numerics α) (red : Reduction) (p : Parameters) (entry : Memory α)
    (b m k : Nat) : α :=
  logicalValue Q red p.width p.eps
    (fun j => entry p.input.allocation (inputOffset p b m j))
    (fun j => entry p.weights.allocation (weightOffset p j)) k

def mathematicalTensor (p : Parameters) (entry : Memory ℝ) (b m k : Nat) : ℝ :=
  mathematical p.width p.eps
    (fun j => entry p.input.allocation (inputOffset p b m j))
    (fun j => entry p.weights.allocation (weightOffset p j)) k

theorem host_outcomes (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry preexisting)
    (valid : ∀ a, rt.allocate = .ok a → p.input.shape.length = 3 →
      p.weights.shape.length = 1 → Domain p) :
    HostSpec p (logicalTensor Q red p entry) rt (executeHost Q red p u rt) := by
  by_cases hx : p.input.shape.length = 3
  · cases ha : rt.allocate with
    | error reason => exact Or.inr (Or.inl ⟨reason,hx,ha,by simp [executeHost,hx,ha]⟩)
    | ok a =>
      by_cases hw : p.weights.shape.length = 1
      · cases hs : rt.setup a with
        | error reason =>
          exact Or.inr (Or.inr (Or.inr (Or.inl
            ⟨a,reason,hx,ha,hw,hs,by simp [executeHost,hx,ha,hw,hs],a.frame⟩)))
        | ok schedule =>
          obtain ⟨final,ran,values,frame,entryFrame⟩ := entry_result Q red p a (valid a ha hx hw) u schedule
          exact Or.inr (Or.inr (Or.inr (Or.inr
            ⟨a,schedule,final,hx,ha,hw,hs,by simp [executeHost,hx,ha,hw,hs,ran],
              a.outputShape,a.outputDtype,a.outputDevice,values,frame,entryFrame⟩)))
      · exact Or.inr (Or.inr (Or.inl
          ⟨a,hx,ha,hw,by simp [executeHost,hx,ha,hw],a.frame⟩))
  · exact Or.inl ⟨hx,by simp [executeHost,hx]⟩

theorem real_host_outcomes (red : Reduction) (p : Parameters) (u : Nat → ℝ)
    (rt : Runtime ℝ p entry preexisting)
    (valid : ∀ a, rt.allocate = .ok a → p.input.shape.length = 3 →
      p.weights.shape.length = 1 → Domain p) :
    HostSpec p (mathematicalTensor p entry) rt (executeHost realNumerics red p u rt) := by
  have equal : logicalTensor realNumerics red p entry = mathematicalTensor p entry := by
    funext b m k
    exact logicalValue_real red p.width p.eps _ _ k
  simpa only [equal] using host_outcomes realNumerics red p u rt valid

/-- Every padded column occurs once; active columns cannot be duplicated by
the chunk/lane enumeration. -/
theorem columns_nodup (p : Parameters) : (columns p).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro t _
    exact (List.nodup_range (n := 4096)).map (fun j k h => by omega)
  · apply (List.nodup_range (n := trips p)).imp
    intro t s different value ht hs
    rcases List.mem_map.mp ht with ⟨i,hi,ei⟩
    rcases List.mem_map.mp hs with ⟨j,hj,ej⟩
    have hib := List.mem_range.mp hi
    have hjb := List.mem_range.mp hj
    apply different
    omega

theorem programGrid_nodup (p : Parameters) : (programGrid p).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro b _
    exact (List.nodup_range (n := p.rows)).map (fun j k h => (Prod.mk.inj h).2)
  · apply (List.nodup_range (n := p.batches)).imp
    intro b c different value hb hc
    rcases List.mem_map.mp hb with ⟨i,_,rfl⟩
    rcases List.mem_map.mp hc with ⟨j,_,equal⟩
    exact different (Prod.mk.inj equal).1.symm

theorem output_bytes_unique (p : Parameters) (unique : OutputUnique p) (base : Nat)
    (b m k b' m' k' lane lane' : Nat)
    (hb : b < p.batches) (hm : m < p.rows) (hk : k < p.width)
    (hb' : b' < p.batches) (hm' : m' < p.rows) (hk' : k' < p.width)
    (hl : lane < 4) (hl' : lane' < 4)
    (equal : base+4*outputOffset p b m k+lane = base+4*outputOffset p b' m' k'+lane') :
    b=b' ∧ m=m' ∧ k=k' ∧ lane=lane' := by
  have offsets : outputOffset p b m k = outputOffset p b' m' k' := by omega
  have same := unique b m k b' m' k' hb hm hk hb' hm' hk' offsets
  exact ⟨same.1,same.2.1,same.2.2,by omega⟩

/-- Machine statements exported with the public outcome theorem. -/
def IndexCorrespondence (p : Parameters) (d : Domain p)
    (f : FiniteAdmission p d preexisting) : Prop :=
  p.width ≤ 2147479552 ∧
  (∀ t, t < trips p →
    (loopBits t).toInt = (t*4096 : Nat) ∧
    (loopBits t+i32 4096).toInt = ((t+1)*4096 : Nat) ∧
    ((loopBits t).slt (i32 p.width) = Bool.true ↔ t*4096 < p.width) ∧
    (loopBits (trips p)).slt (i32 p.width) = Bool.false) ∧
  (∀ b m t j, b < p.batches → m < p.rows → t < trips p → j < 4096 →
    (∀ n ∈ ScalarIntermediates p b m t j, (i32 n).toInt = (n : Int)) ∧
    (columnBits t j).toInt = (t*4096+j : Nat) ∧
    ((columnBits t j).slt (i32 p.width) = Bool.true ↔ t*4096+j < p.width) ∧
    (inputPointer p f.byteBase b m t j).toNat = f.byteBase p.input.allocation+4*inputOffset p b m (t*4096+j) ∧
    (weightPointer p f.byteBase t j).toNat = f.byteBase p.weights.allocation+4*weightOffset p (t*4096+j) ∧
    (outputPointer p f.byteBase b m t j).toNat = f.byteBase p.output.allocation+4*outputOffset p b m (t*4096+j)) ∧
  (∀ b m k, b < p.batches → m < p.rows → k < p.width →
    inputOffset p b m k < d.capacity p.input.allocation ∧
    weightOffset p k < d.capacity p.weights.allocation ∧
    outputOffset p b m k < d.capacity p.output.allocation ∧
    (∀ t ∈ [p.input,p.weights,p.output], ∀ offset, offset < d.capacity t.allocation →
      f.byteBase t.allocation+4*offset+4 ≤ 2^64 ∧ 4 ∣ f.byteBase t.allocation+4*offset) ∧
    (∀ a byte lane, preexisting a → byte < 4*d.capacity a → lane < 4 →
      f.byteBase p.output.allocation+4*outputOffset p b m k+lane ≠ f.byteBase a+byte)) ∧
  (columns p).Nodup ∧ (programGrid p).Nodup

theorem index_correspondence (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting) :
    IndexCorrespondence p d f := by
  refine ⟨(loop_bound_iff p).mp f.loopEnd,loop_exact p d f,?_,
    active_accesses p d f,columns_nodup p,programGrid_nodup p⟩
  intro b m t j hb hm ht hj
  have mask := column_mask_exact p d f t j ht hj
  have ptr := pointers_exact p d f b m t j hb hm ht hj
  exact ⟨intermediates_exact p d f b m t j hb hm ht hj,mask.1,mask.2,ptr.1,ptr.2.1,ptr.2.2⟩

#guard_msgs (drop info) in
#specNonCircular mathematical avoiding [executeHost,entryRun,executeGrid,executeRow,outputPass]
#guard_msgs (drop info) in
#specNonCircular logicalValue avoiding [executeHost,entryRun,executeGrid,executeRow,outputPass]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
