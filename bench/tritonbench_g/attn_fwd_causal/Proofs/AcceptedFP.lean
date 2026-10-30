import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeProjection
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Native
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def WordResult (I : Instructions) (p : Parameters) (initial final : State Word) : Prop :=
  (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
    final.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (outputWord I p initial b h n d hd)) ∧
  (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off)

def ByteResult (I : Instructions) (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (initial : State Word) (before after : Fin pl.arena → BitVec 8) : Prop :=
  (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),∀ k<2,∀ i : Fin pl.arena,
    i.val=pl.base .output+2*logicalAddress p b h n d+k →
      after i=byte (BitVec.ofNat 16 (outputWord I p initial b h n d hd).toNat) k) ∧
  (∀ i : Fin pl.arena,(∀ off,Footprint p off → ∀ k<2,i.val≠pl.base .output+2*off+k) → after i=before i)

theorem finite_result (I : Instructions) (p : Parameters) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (initial final : State Word)
    (binding : WordBinding p storage pl flat initial) (result : WordResult I p initial final) :
    ByteResult I p storage pl initial flat (flattenBytes p storage pl (fun _=>encode16) flat final) := by
  refine ⟨?_,?_⟩
  · intro b hb h hh n hn d hd k hk i hi
    rw [flatten_active p storage pl (fun _=>encode16) flat final (logicalAddress p b h n d) k
      (storage.output b hb h hh n hn d hd) hk i hi,result.1 b hb h hh n hn d hd]
    rfl
  · intro i hi
    exact flatten_frame p storage pl (fun _=>encode16) flat initial final binding.output
      (fun off h=>result.2 p.Out off (Or.inr h)) i hi

/-- Full symbolic execution has no numerical callback. Every result cell is
the concrete fixed-ISA graph at its actual physical output owner. -/
theorem host_native_coverage (p : Parameters) (_guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (alloc : Allocation p abi before)
    (_inputs : RawInputs p before) (pl : Placement p storage) (profile : Profile)
    (_compiler : CompilerBinding p storage pl profile)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi (injectState before)
        (injectAllocation p abi before alloc) work=some result ∧ result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk .fp16
          (answer algebra dotModel p (start (grid p) (n/128,b*p.heads+h,0) (injectState alloc.state)) (lane n d hd))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → result.state.mem r off=(injectState alloc.state).mem r off) :=
  host_result_full (fun _=>algebra) (fun _=>dotModel) p storage abi (injectState before)
    (injectAllocation p abi before alloc) work legal

/-- Public native whole-call result and finite byte frame. Compiler/runtime
realization of the recorded source-derived instruction graph is an explicit
ordinary boundary; the theorem does not assume a whole-output equality. -/
theorem host_native (I : Instructions) (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (alloc : Allocation p abi before)
    (inputs : RawInputs p before) (pl : Placement p storage) (profile : Profile)
    (compiler : CompilerBinding p storage pl profile)
    (flat : Fin pl.arena → BitVec 8) (binding : WordBinding p storage pl flat alloc.state)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi (injectState before)
        (injectAllocation p abi before alloc) work=some result ∧ result.tensor=alloc.output ∧
      WordResult I p alloc.state (projectState I p alloc.state result.state) ∧
      ByteResult I p storage pl alloc.state flat
        (flattenBytes p storage pl (fun _=>encode16) flat (projectState I p alloc.state result.state)) := by
  obtain ⟨result,hr,ht,hv,hf⟩:=host_native_coverage p guard storage abi before alloc inputs pl profile compiler work legal
  have hword : WordResult I p alloc.state (projectState I p alloc.state result.state) := by
    refine ⟨?_,?_⟩
    · intro b hb h hh n hn d hd
      change projectCell I (ownerAt p p.Out (logicalAddress p b h n d))
        (result.state.mem p.Out (logicalAddress p b h n d))=_
      rw [hv b hb h hh n hn d hd]
      simp only [ownerAt,if_pos rfl,featureAt_output p storage b h n d hb hh hn hd]
      rfl
    · intro r off hu
      change projectCell I (ownerAt p r off) (result.state.mem r off)=_
      rw [hf r off hu]
      exact injectState_project I _ alloc.state r off
  exact ⟨result,hr,ht,hword,finite_result I p storage pl flat alloc.state _ binding hword⟩

/-- Deferred numerical equalities are exclusively local scalar operations and
one coordinate of fixed m16n8k8 / m16n8k16 instructions. No whole dot, reduction,
softmax, tensor, result-memory or execution equality is an assumption. -/
theorem fpequiv (actual standard : Instructions)
    (agreement : InstructionAgreement actual standard)
    (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (alloc : Allocation p abi before)
    (inputs : RawInputs p before) (pl : Placement p storage) (profile : Profile)
    (compiler : CompilerBinding p storage pl profile)
    (flat : Fin pl.arena → BitVec 8) (binding : WordBinding p storage pl flat alloc.state)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi (injectState before)
        (injectAllocation p abi before alloc) work=some result ∧ result.tensor=alloc.output ∧
      WordResult standard p alloc.state (projectState actual p alloc.state result.state) ∧
      ByteResult standard p storage pl alloc.state flat
        (flattenBytes p storage pl (fun _=>encode16) flat (projectState actual p alloc.state result.state)) := by
  obtain ⟨result,hr,ht,hword,hbyte⟩:=host_native actual p guard storage abi before alloc inputs pl profile compiler flat binding work legal
  have he : ∀ b h n d (hd : d<128),outputWord actual p alloc.state b h n d hd=outputWord standard p alloc.state b h n d hd := by
    intro b h n d hd
    exact transport id actual.ops standard.ops (agreement.transport actual standard) _
  have hs : WordResult standard p alloc.state (projectState actual p alloc.state result.state) := by
    refine ⟨?_,hword.2⟩
    intro b hb h hh n hn d hd
    rw [hword.1 b hb h hh n hn d hd,he]
  exact ⟨result,hr,ht,hs,finite_result standard p storage pl flat alloc.state _ binding hs⟩

#axiomsClean host_native_coverage
#axiomsClean host_native
#axiomsClean fpequiv
#specNonCircular WordBinding avoiding [invoke,runPrograms,FP.Structural.exec,outputGraph]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Native
