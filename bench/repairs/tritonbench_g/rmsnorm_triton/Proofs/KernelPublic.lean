import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
open scoped Classical
set_option maxHeartbeats 5000000

/-- The kernel launch only. Scheduling enumerates exactly the program grid;
no allocation, setup, wrapper branch or assumed execution result is included. -/
def kernelRun (Q : Numerics α) (red : Reduction) (p : Parameters)
    (undefinedWeight : Nat → α) (schedule : Schedule p) (before : Memory α) : Option (Memory α) :=
  some (executeGrid Q red p undefinedWeight schedule.order before)

/-- Independent ordered recurrence on the actual logical Tensor values. The
compiler tree and scalar FP32 primitives are explicit model parameters. -/
def nativeResult (Q : Numerics α) (red : Reduction) (p : Parameters)
    (x w : FP32.Tensor) (before : Memory α) (b m k : Nat) : α :=
  logicalValue Q red p.width p.eps
    (fun j => tensorRead before x (flatIndex p b m j)) (tensorRead before w) k

/-- Independent full-memory update, specified solely by output coordinates and
logical results. Outside addresses keep the original payload exactly. -/
def referenceMemory (p : Parameters) (f : Nat → Nat → Nat → α)
    (before : Memory α) : Memory α := fun a i =>
  if h : ∃ c : Nat × Nat × Nat, c.1<p.batches ∧ c.2.1<p.rows ∧ c.2.2<p.width ∧
      a=p.output.allocation ∧ i=outputOffset p c.1 c.2.1 c.2.2 then
    f h.choose.1 h.choose.2.1 h.choose.2.2
  else before a i

theorem memory_ext (p : Parameters) (f : Nat → Nat → Nat → α) (before final : Memory α)
    (values : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
      final p.output.allocation (outputOffset p b m k)=f b m k)
    (frame : ∀ a i,Outside p a i → final a i=before a i) :
    final=referenceMemory p f before := by
  funext a i
  unfold referenceMemory
  split
  · rename_i h
    rcases h.choose_spec with ⟨hb,hm,hk,ha,hi⟩
    exact (congrArg₂ final ha hi).trans (values _ _ _ hb hm hk)
  · rename_i h
    exact frame a i (by
      intro b m k hb hm hk same
      exact h ⟨(b,m,k),hb,hm,hk,same⟩)

theorem TensorBinding.input_read (v : TensorBinding x w out p base) (before : Memory α)
    (b m k : Nat) (hb : b<p.batches) (hm : m<p.rows) (hk : k<p.width) :
    tensorRead before v.input (flatIndex p b m k)=before p.input.allocation (inputOffset p b m k) := by
  simp only [tensorRead,v.inputAllocation,v.inputIndex b m k hb hm hk]

theorem TensorBinding.weight_read (v : TensorBinding x w out p base) (before : Memory α)
    (k : Nat) (hk : k<p.width) :
    tensorRead before v.weight k=before p.weights.allocation (weightOffset p k) := by
  simp only [tensorRead,v.weightAllocation,v.weightIndex k hk]

theorem TensorBinding.output_read (v : TensorBinding x w out p base) (before : Memory α)
    (b m k : Nat) (hb : b<p.batches) (hm : m<p.rows) (hk : k<p.width) :
    tensorRead before v.output (flatIndex p b m k)=before p.output.allocation (outputOffset p b m k) := by
  simp only [tensorRead,v.outputAllocation,v.outputIndex b m k hb hm hk]

theorem laneAccumulator_congr (Q : Numerics α) (right : Bool) (width : Nat)
    (x y : Nat → α) (same : ∀ k,k<width → x k=y k) (n : Nat) :
    laneAccumulator Q right width x n=laneAccumulator Q right width y n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    funext j
    simp only [laneAccumulator,ih]
    by_cases active : n*1024+j.val<width
    · rw [if_pos active,if_pos active,same _ active]
    · rw [if_neg active,if_neg active]

theorem logicalValue_congr (Q : Numerics α) (red : Reduction) (width : Nat) (eps : ℝ)
    (x y w v : Nat → α) (same : ∀ k,k<width → x k=y k)
    (weight : ∀ k,k<width → w k=v k) (k : Nat) (hk : k<width) :
    logicalValue Q red width eps x w k=logicalValue Q red width eps y v k := by
  simp only [logicalValue,logicalStd,laneAccumulator_congr Q red.initialRightZero width x y same,
    same k hk,weight k hk]

theorem native_result (Q : Numerics α) (red : Reduction) (p : Parameters)
    (v : TensorBinding x w out p base) (sep : Separation p) (unique : OutputUnique p)
    (u : Nat → α) (schedule : Schedule p) (before : Memory α) :
    ∃ final, kernelRun Q red p u schedule before=some final ∧
      (∀ b m k,b<p.batches → m<p.rows → k<p.width →
        tensorRead final v.output (flatIndex p b m k)=nativeResult Q red p v.input v.weight before b m k) ∧
      final=referenceMemory p (nativeResult Q red p v.input v.weight before) before ∧
      (∀ a i,Outside p a i → final a i=before a i) := by
  obtain ⟨values,frame⟩ := grid_result Q red p sep unique u schedule.order schedule.complete before
  have result : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
      executeGrid Q red p u schedule.order before p.output.allocation (outputOffset p b m k)=
        nativeResult Q red p v.input v.weight before b m k := by
    intro b m k hb hm hk
    rw [values b m k hb hm hk]
    exact logicalValue_congr Q red p.width p.eps _ _ _ _
      (fun j hj => (v.input_read before b m j hb hm hj).symm)
      (fun j hj => (v.weight_read before j hj).symm) k hk
  exact ⟨_,rfl,fun b m k hb hm hk => (v.output_read _ b m k hb hm hk).trans
    (result b m k hb hm hk),memory_ext p _ before _ result frame,frame⟩

theorem nativeResult_real (red : Reduction) (p : Parameters) (x w : FP32.Tensor)
    (before : Memory ℝ) (b m k : Nat) :
    nativeResult realNumerics red p x w before b m k=mathematicalResult p x w before b m k :=
  logicalValue_real red p.width p.eps _ _ k

/-- All active physical addresses of the same actual Tensor objects. No
alignment assumption occurs here; it is derived from the Python checks. -/
def ActualAddresses (p : Parameters) (v : TensorBinding x w out p base) : Prop :=
  ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    v.input.storage.address (v.input.physicalIndex (flatIndex p b m k))=
      v.input.dataPtr.toNat+4*(b*p.input.stride 0+m*p.input.stride 1+k*p.input.stride 2) ∧
    v.weight.storage.address (v.weight.physicalIndex k)=
      v.weight.dataPtr.toNat+4*(k*p.weights.stride 0) ∧
    v.output.storage.address (v.output.physicalIndex (flatIndex p b m k))=
      v.output.dataPtr.toNat+4*(b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2)

theorem actual_addresses (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (base : Nat → Nat) (binding : Binding x w out eps p base)
    (v : TensorBinding x w out p base) : ActualAddresses p v := by
  intro b m k hb hm hk
  have nonempty : x.numel≠0 := by
    rw [binding.numel x w out eps c p base]
    exact Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos (Nat.zero_lt_of_lt hb) (Nat.zero_lt_of_lt hm))
      (Nat.zero_lt_of_lt hk))
  have weightNonempty : w.numel≠0 := by
    simpa [TensorMetadata.numel,c.weightShape] using Nat.ne_of_gt c.positive
  have outputNonempty : out.numel≠0 := by
    simpa [TensorMetadata.numel,binding.outputShape] using nonempty
  have hp := binding.input.pointer nonempty
  have hw := binding.weight.pointer weightNonempty
  have ho := binding.output.pointer outputNonempty
  simp only [FP32.Storage.address,v.inputIndex b m k hb hm hk,v.weightIndex k hk,
    v.outputIndex b m k hb hm hk,v.inputStorage,v.weightStorage,v.outputStorage,
    FP32.Tensor.dataPtr,v.inputMetadata,v.weightMetadata,v.outputMetadata,
    inputOffset,weightOffset,outputOffset]
  constructor
  · omega
  constructor <;> omega

/-- The capacities used by the native access certificate belong to the same
actual backing storage objects; no synthetic larger allocation is substituted. -/
structure StorageBinding (api : MemoryAPI p live)
    (v : TensorBinding x w out p api.byteBase) : Prop where
  input : api.capacity p.input.allocation=v.input.storage.elements
  weight : api.capacity p.weights.allocation=v.weight.storage.elements
  output : api.capacity p.output.allocation=v.output.storage.elements

def ActualStorageSafe (p : Parameters) (v : TensorBinding x w out p base) : Prop :=
  ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    v.input.physicalIndex (flatIndex p b m k)<v.input.storage.elements ∧
    v.weight.physicalIndex k<v.weight.storage.elements ∧
    v.output.physicalIndex (flatIndex p b m k)<v.output.storage.elements ∧
    v.input.storage.address (v.input.physicalIndex (flatIndex p b m k))+4≤2^64 ∧
    v.weight.storage.address (v.weight.physicalIndex k)+4≤2^64 ∧
    v.output.storage.address (v.output.physicalIndex (flatIndex p b m k))+4≤2^64

theorem actual_storage_safe (api : MemoryAPI p live)
    (v : TensorBinding x w out p api.byteBase) (storage : StorageBinding api v) : ActualStorageSafe p v := by
  intro b m k hb hm hk
  have xi := api.inputBounds b m k hb hm hk
  have wi := api.weightBounds k hk
  have yi := api.outputBounds b m k hb hm hk
  rw [storage.input] at xi
  rw [storage.weight] at wi
  rw [storage.output] at yi
  have xs := v.input.storage.spanFits
  have ws := v.weight.storage.spanFits
  have ys := v.output.storage.spanFits
  simp only [v.inputIndex b m k hb hm hk,v.weightIndex k hk,v.outputIndex b m k hb hm hk,
    FP32.Storage.address]
  exact ⟨xi,wi,yi,by omega,by omega,by omega⟩

/-- Checked pointer formation, ABI, active storage bounds, four-byte spans and
allocator frame separation are exported in the same kernel theorem. Empty
leading dimensions require no launch checks and no pointer observation. -/
def MachineResult (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (api : MemoryAPI p live) (binding : Binding x w out eps p api.byteBase)
    (output : x.numel≠0 → LaunchOutput x out) : Prop :=
  ∀ nonempty : x.numel≠0,
    IndexCorrespondence p (binding.domain x w out eps c p api)
      (binding.finite x w out eps c p api nonempty (output nonempty))

theorem machine_result (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (api : MemoryAPI p live) (binding : Binding x w out eps p api.byteBase)
    (output : x.numel≠0 → LaunchOutput x out) : MachineResult x w out eps c p api binding output := by
  intro nonempty
  exact index_correspondence _ _ _

/-- The newly displayed source is definitionally the inherited typed source. -/
theorem kernel_source (X W O : VeriTile.Triton.RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    kernel X W O sxb sxm sxk sw sob som sok N B eps=
      VeriTile.Bench.TritonBenchG.RmsnormTriton.typedKernel .fp32 .fp32 .fp32 Bool.false
        X W O sxb sxm sxk sw sob som sok N B eps := rfl

theorem candidate_source (X W O : VeriTile.Triton.RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps=
      VeriTile.Bench.TritonBenchG.RmsnormTriton.typedKernel .fp32 .fp32 .fp32 Bool.true
        X W O sxb sxm sxk sw sob som sok N B eps := rfl

#axiomsClean native_result
#axiomsClean actual_addresses
#axiomsClean machine_result
#axiomsClean actual_storage_safe
end
end VeriTile.Bench.Repairs.RmsnormTriton
