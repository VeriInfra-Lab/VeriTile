import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelLogical
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelBounds
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelBound
open VeriTile.Triton FP.Structural KernelProgram KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
variable {α β Store Byte : Type} [Inhabited α]

def TensorResults (api : Memory.API β Store Byte) (b : Buffers c s kind seed)
    (p : Program c) (out : Outputs b p) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) (expected : RegionName → Nat → Lane → β) : Prop :=
  let final := Shared.commit api out.toPlacement observe v initial
  let read := fun r i => api.read final ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))
  match kind with
  | .forward => ∀ (t : Fin c.N) (e : Fin 64) (j : Fin 16),
      read "O" (KernelProgram.address c p t.val e.val j.val)=expected "O" t.val (e,j,PUnit.unit)
  | .backward =>
    (∀ (t : Fin (c.N-1)) (e : Fin 64) (j : Fin 16),
      read "DI" (KernelProgram.address c p t.val e.val j.val)=expected "DI" t.val (e,j,PUnit.unit)) ∧
    (∀ t : Fin (c.N-1),read "DG" (gradientAddress c p t.val)=
      expected "DG" t.val (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)) ∧
    (∀ (e : Fin 64) (j : Fin 16),read "DL" (dseedAddress c p e.val j.val)=expected "DL" 0 (e,j,PUnit.unit))

theorem readback (api : Memory.API β Store Byte) (b : Buffers c s kind seed)
    (p : Program c) (out : Outputs b p) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) (r : RegionName) (hr : r∈KernelExecution.outputRoles kind)
    (off : Nat) (active : KernelMemory.footprint kind c p r off) :
    api.read (Shared.commit api out.toPlacement observe v initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex off))=observe r off (v.mem r off) := by
  rw [←out.same r hr]
  exact VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.commit_output api out.toPlacement observe v initial r off active
def InputFrame (api : Memory.API β Store Byte) (b : Buffers c s kind seed)
    (p : Program c) (out : Outputs b p) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) : Prop :=
  ∀ r∈KernelExecution.inputRoles kind seed,∀ i,i<extent b r →
    api.read (Shared.commit api out.toPlacement observe v initial) (KernelTensor.address b r i)=
      api.read initial (KernelTensor.address b r i)
theorem input_frame (api : Memory.API β Store Byte) (b : Buffers c s kind seed)
    (p : Program c) (out : Outputs b p) (isolated : Isolated b p out)
    (observe : RegionName → Nat → Cell α → β) (v : State α) (initial : Store) :
    InputFrame api b p out observe v initial := by
  intro r hr i hi
  apply Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  exact isolated a.1 a.2 ((out.coverage a.1 a.2).mp ha) r hr i hi

theorem unwritten_output_frame (api : Memory.API β Store Byte) (b : Buffers c s kind seed)
    (p : Program c) (out : Outputs b p) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) (r : RegionName)
    (hr : r∈KernelExecution.outputRoles kind) (i : Nat) (hi : i<(b.tensor r).metadata.numel)
    (outside : ¬KernelMemory.footprint kind c p r i) :
    api.read (Shared.commit api out.toPlacement observe v initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))=
      api.read initial ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)) := by
  apply Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  have active := (out.coverage a.1 a.2).mp ha
  apply out.separate a.1 a.2 active r hr i hi
  intro equal
  have first : a.1=r := congrArg Prod.fst equal
  have second : a.2=i := congrArg Prod.snd equal
  exact outside (by simpa only [first,second] using active)

theorem real_tensor (api : Memory.API ℝ Store Byte) (initial : Store)
    (b : Buffers c s kind seed) (currentDevice : Nat) (accepted : KernelLaunch.check b currentDevice=Bool.true)
    (nonempty : 0<c.N) (p : Program c) (out : Outputs b p) (isolated : Isolated b p out) (u : State ℝ)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    KernelLaunch.Admitted b currentDevice ∧ ∃ v,
      KernelExecution.Facts KernelRealReference.algebra kind c s seed p (physical b (api.read initial) u) v ∧
      KernelReal.Outputs kind c s seed p (physical b (api.read initial) u) v ∧
      Shared.TensorImage api out.toPlacement (fun r _ => rawCell kind r) v initial ∧
      InputFrame api b p out (fun r _ => rawCell kind r) v initial ∧
      TensorResults api b p out (fun r _ => rawCell kind r) v initial
        (KernelReal.expected kind c s seed p (physical b (api.read initial) u)) := by
  obtain ⟨v,facts,values⟩ := KernelReal.complete kind c s seed p (physical b (api.read initial) u) nonempty hh hk hv
  refine ⟨(KernelLaunch.check_iff b currentDevice).mp accepted,v,facts,values,
    Shared.complete api out.toPlacement _ v initial,input_frame api b p out isolated _ v initial,?_⟩
  cases kind with
  | forward =>
    intro t e j
    dsimp only
    rw [readback api b p out _ v initial "O" (by simp [KernelExecution.outputRoles]) _ ⟨rfl,t,e,j,rfl⟩,
      values t e j]
    rfl
  | backward =>
    refine ⟨?_,?_,?_⟩
    · intro t e j
      dsimp only
      rw [readback api b p out _ v initial "DI" (by simp [KernelExecution.outputRoles]) _ (Or.inl ⟨rfl,t,e,j,rfl⟩),
        values.1 t e j]
      rfl
    · intro t
      dsimp only
      rw [readback api b p out _ v initial "DG" (by simp [KernelExecution.outputRoles]) _ (Or.inr (Or.inl ⟨rfl,t,rfl⟩)),
        values.2.1 t]
      rfl
    · intro e j
      dsimp only
      rw [readback api b p out _ v initial "DL" (by simp [KernelExecution.outputRoles]) _ (Or.inr (Or.inr ⟨rfl,e,j,rfl⟩)),
        values.2.2 e j]
      rfl

theorem native_tensor (api : Memory.API KernelCarrier.Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (binding : KernelCompiler.Binding platform kind c s seed)
    (b : Buffers c s kind seed) (currentDevice : Nat) (accepted : KernelLaunch.check b currentDevice=Bool.true)
    (abi : Pointers platform binding b) (nonempty : 0<c.N) (p : Program c)
    (out : Outputs b p) (isolated : Isolated b p out) (u : State KernelCarrier.Node)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    KernelLaunch.Admitted b currentDevice ∧ Pointers platform binding b ∧ ∃ v,
      KernelCompiler.Facts platform binding p (physical b (fun a => KernelCarrier.atom (api.read initial a)) u) v ∧
      KernelNative.Outputs (platform.instructions binding.object.primitives) (KernelCompiler.plan platform binding)
        kind c s seed p (physical b (fun a => KernelCarrier.atom (api.read initial a)) u) v ∧
      Shared.TensorImage api out.toPlacement (observe kind) v initial ∧
      InputFrame api b p out (observe kind) v initial ∧
      TensorResults api b p out (observe kind) v initial
        (KernelLogical.expected (platform.instructions binding.object.primitives) (KernelCompiler.plan platform binding)
          b (api.read initial) p) := by
  obtain ⟨v,facts,values⟩ := KernelCompiler.complete platform binding p
    (physical b (fun a => KernelCarrier.atom (api.read initial a)) u) nonempty hh hk hv
  refine ⟨(KernelLaunch.check_iff b currentDevice).mp accepted,abi,v,facts,values,
    Shared.complete api out.toPlacement _ v initial,input_frame api b p out isolated _ v initial,?_⟩
  cases kind with
  | forward =>
    intro t e j
    dsimp only
    rw [readback api b p out _ v initial "O" (by simp [KernelExecution.outputRoles]) _ ⟨rfl,t,e,j,rfl⟩,
      values t e j]
    exact KernelLogical.expected_eq (platform.instructions binding.object.primitives)
      (KernelCompiler.plan platform binding) b (api.read initial) p u "O" t _
  | backward =>
    refine ⟨?_,?_,?_⟩
    · intro t e j
      dsimp only
      rw [readback api b p out _ v initial "DI" (by simp [KernelExecution.outputRoles]) _ (Or.inl ⟨rfl,t,e,j,rfl⟩),
        values.1 t e j]
      exact KernelLogical.expected_eq (platform.instructions binding.object.primitives)
        (KernelCompiler.plan platform binding) b (api.read initial) p u "DI" ⟨t.val,by omega⟩ _
    · intro t
      dsimp only
      rw [readback api b p out _ v initial "DG" (by simp [KernelExecution.outputRoles]) _ (Or.inr (Or.inl ⟨rfl,t,rfl⟩)),
        values.2.1 t]
      exact KernelLogical.expected_eq (platform.instructions binding.object.primitives)
        (KernelCompiler.plan platform binding) b (api.read initial) p u "DG" ⟨t.val,by omega⟩ _
    · intro e j
      dsimp only
      rw [readback api b p out _ v initial "DL" (by simp [KernelExecution.outputRoles]) _ (Or.inr (Or.inr ⟨rfl,e,j,rfl⟩)),
        values.2.2 e j]
      exact KernelLogical.expected_eq (platform.instructions binding.object.primitives)
        (KernelCompiler.plan platform binding) b (api.read initial) p u "DL" ⟨0,nonempty⟩ _

#axiomsClean real_tensor
#axiomsClean native_tensor
#axiomsClean unwritten_output_frame
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelBound
