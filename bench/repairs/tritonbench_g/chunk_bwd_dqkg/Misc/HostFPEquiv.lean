import bench.repairs.tritonbench_g.chunk_bwd_dqkg.HostContract
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Binding
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.DirectNative
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- All returned logical words share one physical placement. The byte buffer
is explicitly materialized from that same symbolic final memory. This is not
a theorem about execution of a flattened floating-point AST or CUDA binary. -/
structure CallResult (observed standard : Instructions) (p : Descriptor) (runtime : Runtime) (c : Config)
    (device : Nat) (A : FlatAlloc) (before : State Word) (alloc : Typed.Allocation before)
    (raw : Nat → BitVec 8) (result : Typed.Result (α:=Stage)) : Prop where
  tensors : result.tensors=physicalDescriptors c device A
  q : ∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
    projectCell observed p runtime (Typed.physicalCell A result.state
      (result.tensors.1.byteBase/4+(h.val*c.T+t.val)*c.K+e.val))=
        .mk .fp32 (outputQ standard p runtime c before h t e)
  k : ∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
    projectCell observed p runtime (Typed.physicalCell A result.state
      (result.tensors.2.1.byteBase/4+(h.val*c.T+t.val)*c.K+e.val))=
        .mk .fp32 (outputK standard p runtime c before h t e)
  g : ∀ (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T),
    projectCell observed p runtime (Typed.physicalCell A result.state
      (result.tensors.2.2.byteBase/4+(k.val*c.heads+h.val)*c.T+t.val))=
        .mk .real (outputG standard p runtime c before k h t)
  cells : ∀ region off,¬gridWrites c (allCTAs c) (region,off) →
    projectCell observed p runtime (result.state.mem region off)=alloc.state.mem region off
  bytes : ∀ i,¬Typed.ByteFootprint c A i →
    Typed.materialize A (graphEncode observed p runtime) raw result.state i=raw i

/-- Concrete profile-indexed source execution, returned physical descriptors,
all logical dQ/dK/per-key-tile dG words, and the full cell/byte frame. No dot,
reduction, output-value or execution equality is an assumption. Native
realization of the recorded ordered graph remains the separately audited
compiler/layout/runtime boundary, with actual-specialization admission; retained profiles are examples. -/
theorem staged_whole_call (I : Instructions) (p : Descriptor) (runtime : Runtime) (c : Config) (device : Nat)
    (A : FlatAlloc) (ha : DenseAllocation c A) (before : State Word)
    (alloc : Typed.Allocation before) (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (raw : Nat → BitVec 8) (hb : Typed.BytesBound A encode raw alloc.state) :
    ∃ result,Typed.invoke (algebra p runtime) dotModel c device A (injectState before)
        (injectAllocation before alloc) order=some result ∧
      CallResult I I p runtime c device A before alloc raw result := by
  obtain ⟨result,hr,ht,hq,hk,hg,hf,hb'⟩:=Typed.whole_call (algebra p runtime) dotModel c device A ha
    (injectState before) (injectAllocation before alloc) order ho (graphEncode I p runtime) raw
    (inject_bytes I p runtime A raw alloc.state hb)
  refine ⟨result,hr,ht,?_,?_,?_,?_,hb'⟩
  · intro h t e
    have hh:=congrArg (projectCell I p runtime) (hq h t e)
    exact hh
  · intro h t e
    exact congrArg (projectCell I p runtime) (hk h t e)
  · intro k h t
    exact congrArg (projectCell I p runtime) (hg k h t)
  · intro r o hu
    rw [hf r o hu]
    exact injectState_project I p runtime alloc.state r o

/-- Local scalar/fixed-instruction agreement lifts to every returned word.
There is no native-to-Real error or numerical accuracy claim. -/
theorem fpequiv (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Descriptor) (runtime : Runtime) (c : Config) (selected : Descriptor) (binding : Binding p c selected)
    (_scale : ScaleBinding runtime c)
    (device : Nat) (A : FlatAlloc) (ha : DenseAllocation c A)
    (before : State Word) (alloc : Typed.Allocation before)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (raw : Nat → BitVec 8) (hb : Typed.BytesBound A encode raw alloc.state) :
    ∃ result,Typed.invoke (algebra selected runtime) dotModel c device A (injectState before)
        (injectAllocation before alloc) order=some result ∧
      CallResult actual standard p runtime c device A before alloc raw result := by
  rw [binding.selectedSyntax]
  obtain ⟨result,hr,h⟩:=staged_whole_call actual p runtime c device A ha before alloc order ho raw hb
  have eq (g : Graph) : g.eval actual.ops=g.eval standard.ops :=
    transport id actual.ops standard.ops (agreement.transport actual standard) g
  refine ⟨result,hr,h.tensors,?_,?_,?_,h.cells,h.bytes⟩
  · intro hd t e
    rw [h.q hd t e]
    exact congrArg (Cell.mk .fp32) (eq _)
  · intro hd t e
    rw [h.k hd t e]
    exact congrArg (Cell.mk .fp32) (eq _)
  · intro k hd t
    rw [h.g k hd t]
    exact congrArg (Cell.mk .real) (eq _)

#axiomsClean staged_whole_call
#axiomsClean fpequiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.DirectNative
open VeriTile Triton Whole Native Direct
noncomputable section

/-- Public direct wrapper endpoint. The only numerical premise concerns scalar
instructions and one output coordinate of fixed m16n8k8 TF32. Compiler/profile
mapping remains explicit. Allocation, layout, scale, source safety, whole
execution, physical input decoding, every output word and byte frame compose. -/
theorem chunk_bwd_dqkg_if_local (actual standard : Instructions)
    (agreement : InstructionAgreement actual standard)
    (p : Metadata) (h : checkInputs p=.ok ()) (a : Direct.Allocation p)
    (outputs : checkOutputs p a=.ok ()) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (chosen : Fin 2) (heap : Heap) :
    ∃ returned,wrapper actual p a mapping choices chosen heap=.ok returned ∧
      Certificate actual standard p a mapping choices chosen heap returned := by
  refine ⟨result actual p a mapping choices chosen heap,?_,?_⟩
  · simp [wrapper,h,outputs] <;> rfl
  · exact complete actual standard agreement p ((checkInputs_iff p).mp h) a
      ((checkOutputs_iff p a).mp outputs) mapping choices chosen heap

/-- Native source conformance without a cross-implementation numerical law. -/
theorem chunk_bwd_dqkg_native (actual : Instructions)
    (p : Metadata) (h : checkInputs p=.ok ()) (a : Direct.Allocation p)
    (outputs : checkOutputs p a=.ok ()) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (chosen : Fin 2) (heap : Heap) :
    ∃ returned,wrapper actual p a mapping choices chosen heap=.ok returned ∧
      Certificate actual actual p a mapping choices chosen heap returned := by
  apply chunk_bwd_dqkg_if_local actual actual
    ⟨fun _=>rfl,fun _ _=>rfl,fun _ _=>rfl,fun _ _=>rfl,fun _ _=>rfl,
      fun _=>rfl,fun _ _ _=>rfl,fun _ _ _ _ _=>rfl⟩ p h a outputs mapping choices chosen heap

#axiomsClean chunk_bwd_dqkg_if_local
#axiomsClean chunk_bwd_dqkg_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.DirectNative
