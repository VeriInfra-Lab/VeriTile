import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16NativeScale
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.StorageGuard
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Configuration
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16NativeOperands
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
open VeriTile Triton FP.Structural Source Whole VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def WordResult (I : Instructions) (p : Parameters) (selected : Selected p) (initial final : State Word) : Prop :=
  (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
    final.mem p.output (logicalAddress p b h n d)=.mk .bf16 (outputWord I p selected initial b h n d hd)) ∧
  (∀ r off,OutsideLogical p r off → final.mem r off=initial.mem r off)

def ByteResult (I : Instructions) (p : Parameters) (selected : Selected p) (storage : TensorStorage p) (pl : Placement p storage)
    (initial : State Word) (before after : Fin pl.arena → BitVec 8) : Prop :=
  (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),∀ k<2,∀ i : Fin pl.arena,
    i.val=pl.base .output+2*logicalAddress p b h n d+k →
      after i=byte (BitVec.ofNat 16 (outputWord I p selected initial b h n d hd).toNat) k) ∧
  (∀ i : Fin pl.arena,(∀ off,Footprint p off → ∀ k<2,i.val≠pl.base .output+2*off+k) → after i=before i)

theorem finite_result (I : Instructions) (p : Parameters) (selected : Selected p) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (initial final : State Word)
    (binding : WordBinding p storage pl flat initial) (result : WordResult I p selected initial final) :
    ByteResult I p selected storage pl initial flat (flattenBytes p storage pl (fun _=>encode16) flat final) := by
  refine ⟨?_,?_⟩
  · intro b hb h hh n hn d hd k hk i hi
    rw [flatten_active p storage pl (fun _=>encode16) flat final (logicalAddress p b h n d) k
      (storage.output b hb h hh n hn d hd) hk i hi,result.1 b hb h hh n hn d hd]
    rfl
  · intro i hi
    exact flatten_frame p storage pl (fun _=>encode16) flat initial final binding.output
      (fun off h=>result.2 p.output off ((outside_iff p p.output off).mpr (Or.inr h))) i hi

/-- Symbolic full-grid execution with the concrete seeded MMA and reduction
plans. Inputs range over all half words, including NaNs and infinities. -/
theorem host_native_coverage (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (_inputs : RawInputs p before)
    (pl : Placement p storage) (profile : Profile) (_compiler : CompilerBinding p storage abi pl profile)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        result.state.mem p.output (logicalAddress p b h n d)=.mk .bf16
          (answer (modelFor p) p (start (grid p) (n/p.blockM,b*p.heads+h,0) (injectState before)) (lane p guard.selected n d hd))) ∧
      (∀ r off,OutsideLogical p r off → result.state.mem r off=(injectState before).mem r off) :=
  host_result (fun _=>algebraFor p) (fun _=>modelFor p) p guard storage abi (injectState before) work legal

/-- Whole caller-owned Out result, unchanged descriptor, Python None, and
finite byte frame. Compiler/runtime graph realization remains an ordinary
boundary; no tensor, whole-reduction or whole-dot numerical premise occurs. -/
theorem host_native (I : Instructions) (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (pl : Placement p storage) (profile : Profile) (compiler : CompilerBinding p storage abi pl profile)
    (flat : Fin pl.arena → BitVec 8) (binding : WordBinding p storage pl flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult I p guard.selected before (projectState I p before result.state) ∧
      ByteResult I p guard.selected storage pl before flat
        (flattenBytes p storage pl (fun _=>encode16) flat (projectState I p before result.state)) := by
  obtain ⟨result,hr,ht,hnone,hv,hf⟩:=host_native_coverage p guard storage abi before inputs pl profile compiler work legal
  have hword : WordResult I p guard.selected before (projectState I p before result.state) := by
    refine ⟨?_,?_⟩
    · intro b hb h hh n hn d hd
      change projectCell I (ownerAt p p.output (logicalAddress p b h n d))
        (result.state.mem p.output (logicalAddress p b h n d))=_
      rw [hv b hb h hh n hn d hd]
      simp only [ownerAt,if_pos rfl,featureAt_output p guard b h n d hb hh hn hd]
      rfl
    · intro r off hu
      change projectCell I (ownerAt p r off) (result.state.mem r off)=_
      rw [hf r off hu]
      exact injectState_project I _ before r off
  exact ⟨result,hr,ht,hnone,hword,finite_result I p guard.selected storage pl flat before _ binding hword⟩

/-- Numerical assumptions are exactly scalar instructions and one output
coordinate of fixed m16n8k16 instructions, with their actual incoming seed. -/
theorem fpequiv (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (pl : Placement p storage) (profile : Profile) (compiler : CompilerBinding p storage abi pl profile)
    (flat : Fin pl.arena → BitVec 8) (binding : WordBinding p storage pl flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult standard p guard.selected before (projectState actual p before result.state) ∧
      ByteResult standard p guard.selected storage pl before flat
        (flattenBytes p storage pl (fun _=>encode16) flat (projectState actual p before result.state)) := by
  obtain ⟨result,hr,ht,hnone,hword,hbyte⟩:=host_native actual p guard storage abi before inputs pl profile compiler flat binding work legal
  have he : ∀ b h n d (hd : d<p.head),outputWord actual p guard.selected before b h n d hd=outputWord standard p guard.selected before b h n d hd := by
    intro b h n d hd
    exact transport id actual.ops standard.ops (agreement.transport actual standard) _
  have hs : WordResult standard p guard.selected before (projectState actual p before result.state) := by
    refine ⟨?_,hword.2⟩
    intro b hb h hh n hn d hd
    rw [hword.1 b hb h hh n hn d hd,he]
  exact ⟨result,hr,ht,hnone,hs,finite_result standard p guard.selected storage pl flat before _ binding hs⟩

#axiomsClean host_native_coverage
#axiomsClean host_native
#axiomsClean fpequiv
#specNonCircular WordBinding avoiding [invoke,runPrograms,FP.Structural.exec,outputGraph]

/-- Public current-storage composition: all source loads come from the current
finite caller bytes, all CTAs execute, and the projected result mutates exactly
the logical output footprint while preserving the descriptor and Python None.
Only the local instruction agreement remains a numerical premise. -/
theorem fpequiv_current (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (pl : Placement p storage) (profile : Profile) (compiler : CompilerBinding p storage abi pl profile)
    (flat : Fin pl.arena → BitVec 8) (binding : WordBinding p storage pl flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    (∀ job∈work,CurrentOperands p storage pl flat (start (grid p) job before)) ∧
    ∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult standard p guard.selected before (projectState actual p before result.state) ∧
      ByteResult standard p guard.selected storage pl before flat
        (flattenBytes p storage pl (fun _=>encode16) flat (projectState actual p before result.state)) := by
  refine ⟨?_,fpequiv actual standard agreement p guard storage abi before inputs pl profile compiler flat binding work legal⟩
  intro job hj
  exact grid_current_operands p guard storage pl flat before binding inputs job (legal.mem_iff.mpr hj)

#axiomsClean fpequiv_current
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
