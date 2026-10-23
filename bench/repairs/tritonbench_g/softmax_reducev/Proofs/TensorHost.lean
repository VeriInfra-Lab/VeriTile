import bench.repairs.tritonbench_g.softmax_reducev.Proofs.Admission
import bench.repairs.tritonbench_g.softmax_reducev.Proofs.Completion
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Host

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.Checked
open VeriTile.Triton FP.Structural
open SoftmaxTriton1 (Result)
noncomputable section
abbrev Object := SoftmaxTriton1.Contiguous.Object

/-- Ordinary contiguous signed-int32 storage API binding. The shared metadata
contains dtype/layout/device/shape/contiguity and the queried data pointer. -/
structure MetadataObject where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storage : VeriTile.Torch.FP32.Storage
  storageOffset : Nat
  pointer : 0 < metadata.numel → metadata.dataPtr=BitVec.ofNat 64 (storage.address storageOffset)
  contiguous : metadata.isContiguous=Bool.true → ∀ i, i < metadata.numel →
    storageOffset+i < storage.elements

def payloadObservation (x : Object) : Observation :=
  ⟨x.metadata,[],x.tensor.storage,x.tensor.storageOffset,x.tensor.object⟩
def MetadataObject.observation (x : MetadataObject) : Observation :=
  ⟨x.metadata,[],x.storage,x.storageOffset,x.object⟩

structure TensorCall where
  (logics values output : Object)
  (locations starts lengths : MetadataObject)
  maximum : Nat

def TensorCall.observation (c : TensorCall) : Call :=
  ⟨payloadObservation c.logics,payloadObservation c.values,payloadObservation c.output,
    c.locations.observation,c.starts.observation,c.lengths.observation,c.maximum⟩

abbrev storageRegion := TokenSoftmaxLlama.Checked.storageRegion

def view {α : Type} (c : TensorCall) (m : Host.Memory α) : Host.Memory α := fun r i =>
  if r="Logics" then m c.logics.region (c.logics.tensor.storageOffset+i)
  else if r="V" then m c.values.region (c.values.tensor.storageOffset+i)
  else if r="Out" then m c.output.region (c.output.tensor.storageOffset+i)
  else if r="Locations" then m (storageRegion c.locations.storage) (c.locations.storageOffset+i)
  else if r="Starts" then m (storageRegion c.starts.storage) (c.starts.storageOffset+i)
  else if r="Lengths" then m (storageRegion c.lengths.storage) (c.lengths.storageOffset+i)
  else m r i

def physical {α : Type} (c : TensorCall) (before after : Host.Memory α) : Host.Memory α :=
  fun r i => if r=c.output.region ∧ c.output.tensor.storageOffset ≤ i then
    after "Out" (i-c.output.tensor.storageOffset) else before r i

theorem physical_read {α : Type} (c : TensorCall) (before after : Host.Memory α) (i : Nat) :
    physical c before after c.output.region (c.output.tensor.storageOffset+i)=after "Out" i := by
  simp [physical]

def Exterior (c : TensorCall) (r : RegionName) (i : Nat) :=
  r≠c.output.region ∨ i < c.output.tensor.storageOffset ∨
    Host.Off (host c.observation) (Host.workset (host c.observation)) (i-c.output.tensor.storageOffset)

theorem physical_frame {α : Type} (c : TensorCall) (before after : Host.Memory α)
    (frame : ∀ r i,Host.Outside (host c.observation) 0 r i → after r i=view c before r i) :
    ∀ r i,Exterior c r i → physical c before after r i=before r i := by
  intro r i outside
  unfold physical
  split
  · rename_i hit
    rcases hit with ⟨rfl,lower⟩
    rw [frame "Out" (i-c.output.tensor.storageOffset) (by
      apply Or.inr;apply Or.inr
      rcases outside with off|off|off
      · exact False.elim (off rfl)
      · omega
      · simpa using off)]
    simp [view,Nat.add_sub_of_le lower]
  · rfl

/-- A local call executes its complete source evaluator. There is no numerical
or final-output callback: these calls are instantiated below by concrete ASTs. -/
def sourceCall {α : Type} (evaluate : Host.Job → (Nat → α) → Option (Nat → α))
    (j : Host.Job) (s : Nat → α) : Result (Nat → α) Unit :=
  match evaluate j s with
  | some t => .ok () t
  | none => .error "source execution" s

theorem source_runtime {α : Type} (c : Host.Call)
    (evaluate : Host.Job → (Nat → α) → Option (Nat → α)) :
    Host.RuntimeGeneral c evaluate (sourceCall evaluate) := by
  constructor
  · intro j s t legal run
    cases eq : evaluate j s with
    | none => simp [sourceCall,eq] at run
    | some u =>
      have same : u=t := by simpa [sourceCall,eq] using run
      simpa [eq,same]
  · intro j s e t legal run
    cases eq : evaluate j s with
    | some u => simp [sourceCall,eq] at run
    | none =>
      have both : "source execution"=e ∧ s=t := by simpa [sourceCall,eq] using run
      have same := both.2
      subst t
      exact fun _ _ => rfl

theorem source_jobs_total {α : Type} (evaluate : Host.Job → (Nat → α) → Option (Nat → α))
    (total : ∀ j s,∃ t,evaluate j s=some t) (work : List Host.Job) (s : Nat → α) :
    ∃ t,Host.runJobs (sourceCall evaluate) work s=.ok () t := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons j rest ih =>
    obtain ⟨u,run⟩ := total j s
    obtain ⟨t,tail⟩ := ih u
    exact ⟨t,by simp [Host.runJobs,sourceCall,run,Result.bind,tail]⟩

theorem source_whole {α : Type} [Inhabited α] (c : Host.Call)
    (evaluate : Host.Job → (Nat → α) → Option (Nat → α)) (target : Host.Job → Nat → α)
    (source : Host.SourceLawGeneral c evaluate target) (total : ∀ j s,∃ t,evaluate j s=some t)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) (m : Host.Memory α) :
    ∃ final,Host.wrapper c 0 work (sourceCall evaluate) m=.ok () final ∧
      (∀ j,Host.Legal c j → ∀ d,d < c.dimensions → final "Out" (Host.address c j d)=target j d) ∧
      (∀ r i,Host.Outside c 0 r i → final r i=m r i) := by
  by_cases empty : Host.noWork c
  · refine ⟨m,by simp [Host.wrapper,empty],?_,fun _ _ _ => rfl⟩
    intro j legal d hd
    rcases empty with a|b <;> dsimp [Host.Legal] at legal <;> omega
  · obtain ⟨out,run⟩ := source_jobs_total evaluate total work (Host.readOutput 0 m)
    have runFull : Host.wrapper c 0 work (sourceCall evaluate) m=.ok () (Host.putOutput 0 m out) := by
      simp [Host.wrapper,empty,Host.runFull_projection,run,Host.liftResult]
    have result := Host.whole_general c evaluate target source 0 (sourceCall evaluate)
      (source_runtime c evaluate) work complete m _ runFull
    exact ⟨_,runFull,by simpa [Host.address,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using result.1,result.2⟩

structure TensorResult (α : Type) where
  output : VeriTile.Torch.FP32.Tensor
  returned : Option Nat
  memory : Host.Memory α

def liftResult {α : Type} (c : TensorCall) (before : Host.Memory α) :
    Result (Host.Memory α) Unit → Result (Host.Memory α) (TensorResult α)
  | .ok () after => .ok ⟨c.output.tensor,none,physical c before after⟩ (physical c before after)
  | .error e after => .error e (physical c before after)

def invoke {α : Type} (c : TensorCall) (work : List Host.Job)
    (evaluate : Host.Job → (Nat → α) → Option (Nat → α)) (m : Host.Memory α) :=
  liftResult c m (Host.wrapper (host c.observation) 0 work (sourceCall evaluate) (view c m))

#axiomsClean physical_frame
#axiomsClean source_jobs_total
#axiomsClean source_whole
end
end VeriTile.Bench.Repairs.SoftmaxReduceV.Checked
