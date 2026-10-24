import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReal
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCoverage
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelIndex
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelBound
open VeriTile.Triton FP.Structural KernelInputs KernelInputs.Shared KernelNative.Shared
namespace Storage
export VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
  (Placement TensorImage commit attach nativeAttach complete)
end Storage
noncomputable section

def forwardFootprint (p : ForwardParameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  ¬Forward.Outside p s r o
def backwardFootprint (p : BackwardParameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  ¬KernelFrame.Outside p s r o

theorem real_forward (rest : Algebra ℝ) (p : ForwardParameters) (read : Nat → ℝ)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ)
    (h : KernelExecution.ForwardAdmissible p) (hb : p.columns≤p.block) :
    Pointers views ∧ ∃ t,
      KernelExecution.ForwardFacts (Logical.algebra rest) p (physical views u) t ∧
      KernelReal.ForwardOutputs p (logical views u) t := by
  rw [physical_eq_logical]
  exact ⟨pointers views,KernelReal.forward rest p (logical views u) h hb⟩

theorem real_backward (rest : Algebra ℝ) (p : BackwardParameters) (read : Nat → ℝ)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ)
    (h : KernelReadback.Admissible p) (hb : p.columns≤p.block) :
    Pointers views ∧ ∃ t,
      KernelExecution.BackwardFacts (Logical.algebra rest) p (physical views u) t ∧
      KernelReal.BackwardOutputs p (logical views u) t := by
  rw [physical_eq_logical]
  exact ⟨pointers views,KernelReal.backward rest p (logical views u) h hb⟩

theorem native_forward (platform : KernelCompiler.Platform) (p : ForwardParameters) (host : UInt64)
    (R : Reductions) (S : OutputSchedule) (binding : KernelCompiler.Binding platform (.inl p) host R S)
    (conversion : KernelCompiler.Conversion platform (.inl p) host binding)
    (read : Nat → Term) (views : RegionName → Option (ReadView Term read)) (u : State Term)
    (h : KernelExecution.ForwardAdmissible p) :
    (Pointers views ∧ KernelNative.CountFacts (platform.primitives binding.object.primitives) p.columns ∧
      ∀ schedule,eval (platform.primitives binding.object.primitives) schedule
      (.literal (some .fp32) .real p.epsilon)=platform.epsilon binding.object host) ∧ ∃ t,
      KernelExecution.ForwardFacts (algebra R) p (physical views u) t ∧
      KernelNative.ForwardOutputs R (platform.primitives binding.object.primitives)
        (withEpsilon (platform.primitives binding.object.primitives) p.epsilon (platform.epsilon binding.object host))
        S p (logical views u) t := by
  rw [physical_eq_logical]
  obtain ⟨hc,hs,t,ht⟩ := KernelCompiler.forward platform p host R S binding conversion (logical views u) h
  exact ⟨⟨pointers views,hc,hs⟩,t,ht⟩

theorem native_backward (platform : KernelCompiler.Platform) (p : BackwardParameters) (host : UInt64)
    (R : Reductions) (S : OutputSchedule) (binding : KernelCompiler.Binding platform (.inr p) host R S)
    (conversion : KernelCompiler.CountConversion platform (.inr p) host binding)
    (read : Nat → Term) (views : RegionName → Option (ReadView Term read)) (u : State Term)
    (h : KernelReadback.Admissible p) :
    (Pointers views ∧ KernelNative.CountFacts (platform.primitives binding.object.primitives) p.columns) ∧ ∃ t,
      KernelExecution.BackwardFacts (algebra R) p (physical views u) t ∧
      KernelNative.BackwardOutputs R (platform.primitives binding.object.primitives) S p (logical views u) t := by
  rw [physical_eq_logical]
  obtain ⟨hc,t,ht⟩ := KernelCompiler.backward platform p host R S binding conversion (logical views u) h
  exact ⟨⟨pointers views,hc⟩,t,ht⟩

/-- Fresh output allocations and live input storage have disjoint bytes.
Read/read sharing is unrestricted. This is an allocation layout condition. -/
def Fresh {α : Type} {read : Nat → α} (views : RegionName → Option (ReadView α read))
    (outputs : Storage.Placement active) : Prop :=
  ∀ r i,active r i → ∀ q v,views q=some v → ∀ j,j<v.tensor.metadata.numel →
    VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.Apart (outputs.address r i)
      ((VeriTile.Torch.FP32.dataView v.tensor v.laws (v.checked.dense rfl)).address j)

def InputFrame {α β Store Byte : Type} [Inhabited α] {read : Nat → α}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API β Store Byte)
    (views : RegionName → Option (ReadView α read)) (outputs : Storage.Placement active)
    (observe : RegionName → Nat → Cell α → β) (t : State α) (initial : Store) : Prop :=
  ∀ q v,views q=some v → ∀ j,j<v.tensor.metadata.numel →
    api.read (Storage.commit api outputs observe t initial)
      ((VeriTile.Torch.FP32.dataView v.tensor v.laws (v.checked.dense rfl)).address j)=
    api.read initial ((VeriTile.Torch.FP32.dataView v.tensor v.laws (v.checked.dense rfl)).address j)

theorem input_frame {α β Store Byte : Type} [Inhabited α] {read : Nat → α}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API β Store Byte)
    (views : RegionName → Option (ReadView α read)) (outputs : Storage.Placement active)
    (observe : RegionName → Nat → Cell α → β) (t : State α) (initial : Store)
    (fresh : Fresh views outputs) : InputFrame api views outputs observe t initial := by
  intro q v hv j hj
  apply VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  exact fresh a.1 a.2 ((outputs.coverage a.1 a.2).mp ha) q v hv j hj

theorem attachFresh {α β Store Byte : Type} [Inhabited α] {read : Nat → α}
    {H : Prop} {P : State α → Prop} (source : H ∧ ∃ t,P t)
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API β Store Byte)
    (views : RegionName → Option (ReadView α read)) (outputs : Storage.Placement active)
    (observe : RegionName → Nat → Cell α → β) (initial : Store) (fresh : Fresh views outputs) :
    H ∧ ∃ t,P t ∧ Storage.TensorImage api outputs observe t initial ∧
      InputFrame api views outputs observe t initial := by
  obtain ⟨h,t,ht⟩ := source
  exact ⟨h,t,ht,Storage.complete api outputs observe t initial,input_frame api views outputs observe t initial fresh⟩

theorem nativeAttachFresh {α Store Byte : Type} [Inhabited α] {read : Nat → α}
    {H : Prop} {P : State α → Prop} (source : H ∧ ∃ t,P t)
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte)
    (views : RegionName → Option (ReadView α read)) (outputs : Storage.Placement active)
    (observe : RegionName → Nat → Cell α → Word) (initial : Store) (fresh : Fresh views outputs) :
    H ∧ ∃ t,P t ∧ Storage.TensorImage api outputs observe t initial ∧
      InputFrame api views outputs observe t initial ∧
      (∀ r i,active r i →
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (Storage.commit api outputs observe t initial) a⟩ : VeriTile.Torch.FP32.Word))
          (outputs.tensor r) (outputs.laws r) i).bits=observe r i (t.mem r i)) := by
  obtain ⟨h,t,ht,hi,hl⟩ := Storage.nativeAttach source api outputs observe initial
  exact ⟨h,t,ht,hi,input_frame api views outputs observe t initial fresh,hl⟩

def real_forward_tensor {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API ℝ Store Byte) (initial : Store)
    (rest : Algebra ℝ) (c : Admission.ForwardCall) (check : Admission.forwardCheck c=Bool.true)
    (inputs : ForwardBinding c) (u : State ℝ) (rowBound : u.pids 0<c.common.rows)
    (outputs : Storage.Placement (forwardFootprint (KernelDispatch.forward c) u))
    (fresh : Fresh (forwardViews (api.read initial) c inputs ((Admission.forwardCheck_iff c).mp check)) outputs) :=
  And.intro (forward_coverage (api.read initial) c inputs ((Admission.forwardCheck_iff c).mp check) (u.pids 0) rowBound)
  (attachFresh
    (real_forward rest (KernelDispatch.forward c) (api.read initial)
      (forwardViews (api.read initial) c inputs ((Admission.forwardCheck_iff c).mp check)) u
      (KernelDispatch.forward_admissible c) (KernelDispatch.checked_forward_block c check))
    api (forwardViews (api.read initial) c inputs ((Admission.forwardCheck_iff c).mp check))
    outputs (fun _ _ cell => cell.read .fp32) initial fresh)

def real_backward_tensor {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API ℝ Store Byte) (initial : Store)
    (rest : Algebra ℝ) (c : Admission.BackwardCall) (check : Admission.backwardCheck c=Bool.true)
    (inputs : BackwardBinding c) (u : State ℝ) (programBound : u.pids 0<c.programs)
    (outputs : Storage.Placement (backwardFootprint (KernelDispatch.backward c) u))
    (fresh : Fresh (backwardViews (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check)) outputs) :=
  And.intro (backward_parameters (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check))
  (And.intro programBound (And.intro
    (fun row hr => backward_coverage (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check) row hr)
  (attachFresh
    (real_backward rest (KernelDispatch.backward c) (api.read initial)
      (backwardViews (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check)) u
      (KernelDispatch.backward_admissible c) (KernelDispatch.checked_backward_block c check))
    api (backwardViews (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check))
    outputs (fun _ _ cell => cell.read .fp32) initial fresh)))

def words (read : Nat → Word) : Nat → Term := fun a => .leaf "physical" a (read a)

def native_forward_tensor {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (c : Admission.ForwardCall) (check : Admission.forwardCheck c=Bool.true)
    (host : UInt64) (R : Reductions) (S : OutputSchedule)
    (binding : KernelCompiler.Binding platform (.inl (KernelDispatch.forward c)) host R S)
    (conversion : KernelCompiler.Conversion platform (.inl (KernelDispatch.forward c)) host binding)
    (inputs : ForwardBinding c) (u : State Term) (rowBound : u.pids 0<c.common.rows)
    (outputs : Storage.Placement (forwardFootprint (KernelDispatch.forward c) u))
    (fresh : Fresh (forwardViews (words (api.read initial)) c inputs ((Admission.forwardCheck_iff c).mp check)) outputs) :=
  And.intro (forward_coverage (words (api.read initial)) c inputs ((Admission.forwardCheck_iff c).mp check) (u.pids 0) rowBound)
  (nativeAttachFresh
    (native_forward platform (KernelDispatch.forward c) host R S binding conversion (words (api.read initial))
      (forwardViews (words (api.read initial)) c inputs ((Admission.forwardCheck_iff c).mp check)) u
      (KernelDispatch.forward_admissible c))
    api (forwardViews (words (api.read initial)) c inputs ((Admission.forwardCheck_iff c).mp check))
    outputs (fun r i => KernelNative.observe (platform.primitives binding.object.primitives) (S r i) c.common.columns) initial fresh)

def native_backward_tensor {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (c : Admission.BackwardCall) (check : Admission.backwardCheck c=Bool.true)
    (host : UInt64) (R : Reductions) (S : OutputSchedule)
    (binding : KernelCompiler.Binding platform (.inr (KernelDispatch.backward c)) host R S)
    (conversion : KernelCompiler.CountConversion platform (.inr (KernelDispatch.backward c)) host binding)
    (inputs : BackwardBinding c) (u : State Term) (programBound : u.pids 0<c.programs)
    (outputs : Storage.Placement (backwardFootprint (KernelDispatch.backward c) u))
    (fresh : Fresh (backwardViews (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check)) outputs) :=
  And.intro (backward_parameters (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check))
  (And.intro programBound (And.intro
    (fun row hr => backward_coverage (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check) row hr)
  (nativeAttachFresh
    (native_backward platform (KernelDispatch.backward c) host R S binding conversion (words (api.read initial))
      (backwardViews (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check)) u
      (KernelDispatch.backward_admissible c))
    api (backwardViews (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check))
    outputs (fun r i => KernelNative.observe (platform.primitives binding.object.primitives) (S r i) c.common.columns) initial fresh)))

#axiomsClean real_forward
#axiomsClean real_backward
#axiomsClean native_forward
#axiomsClean native_backward
#axiomsClean real_forward_tensor
#axiomsClean real_backward_tensor
#axiomsClean native_forward_tensor
#axiomsClean native_backward_tensor
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelBound
