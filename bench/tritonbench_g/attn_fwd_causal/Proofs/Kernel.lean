import bench.tritonbench_g.attn_fwd_causal.Proofs.AcceptedReal
import bench.tritonbench_g.attn_fwd_causal.Proofs.AcceptedFP
import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeReferenceBinding
import bench.tritonbench_g.attn_fwd_causal.Contract

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Kernel
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

/-- The existing checked input domain, supplied by the Python assertions. -/
abbrev InputContract := Contract.InputContract

/-- Ordinary metadata of the actual output supplied by empty_like. This binds
the kernel's output region and passed strides to its Tensor coordinates; it
does not prescribe any initial or final output value. -/
structure OutputBinding (p : Parameters) (abi : InputABI p) (out : Descriptor) : Prop where
  region : out.region=p.Out
  shape : out.shape=[p.batches,p.heads,p.length,128]
  strides : 0<p.batches → 0<p.heads → 0<p.length → out.strides=[p.o.batch,p.o.head,p.o.row,p.o.dim]
  dtype : out.dtype=.fp16
  device : out.device=abi.query.device
  ordinary : out.negative=Bool.false ∧ out.conjugate=Bool.false

def tensorOffset (out : Descriptor) (b h n d : Nat) : Nat :=
  match out.strides with
  | [sz,sh,sn,sd]=>b*sz+h*sh+n*sn+d*sd
  | _=>0

theorem output_offset (p : Parameters) (abi : InputABI p) (out : Descriptor)
    (binding : OutputBinding p abi out) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) :
    tensorOffset out b h n d=logicalAddress p b h n d := by
  simp only [tensorOffset,binding.strides (by omega) (by omega) (by omega),logicalAddress]

/-- Reuse the existing full-source and grid proof directly at the pre-kernel
memory boundary. Every CTA executes the shared quotation against current
memory; no wrapper invocation or allocator model is part of this execution. -/
theorem grid_result {α : Type} [Inhabited α]
    (M : Program → Algebra α) (dot : Program → Dot α)
    (p : Parameters) (storage : TensorStorage p) (initial : State α)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms M dot p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16
          (answer (M (n/128,b*p.heads+h,0)) (dot (n/128,b*p.heads+h,0)) p
            (start (grid p) (n/128,b*p.heads+h,0) initial) (lane n d hd))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off) := by
  have unique : Unique p (grid p) work initial := by
    intro a ha b hb i j hi hj he
    exact grid_unique p storage initial a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
  obtain ⟨final,hr,hv,hf⟩:=run_correct M dot p (separated p storage) (grid p) work initial unique
  refine ⟨final,hr,?_,?_⟩
  · intro b hb h hh n hn d hd
    have active : Source.active p (start (grid p) (n/128,b*p.heads+h,0) initial) (lane n d hd) :=
      ⟨by change n/128*128+n%128<p.length;omega,hd⟩
    have value:=hv _ (legal.mem_iff.mp (grid_covers p b h n d hb hh hn hd).1) _ active
    rw [physical_address p initial b h n d hh hd] at value
    exact value
  · intro r off outside
    apply hf r off
    rcases outside with different|outside
    · exact Or.inl different
    · right
      intro pid member i active equal
      exact outside (equal ▸ active_footprint p initial pid (legal.mem_iff.mpr member) i active)

theorem real_result (p : Parameters) (storage : TensorStorage p) (initial : State Real.R)
    (μ : RegionName → Nat → ℝ) (reads : Real.Reads p initial μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off) := by
  obtain ⟨final,run,values,frame⟩:=grid_result (fun _=>Real.algebra) (fun _=>Real.dotModel) p storage initial work legal
  refine ⟨final,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd]
  have active : Source.active p (start (grid p) (n/128,b*p.heads+h,0) initial) (lane n d hd) :=
    ⟨by change n/128*128+n%128<p.length;omega,hd⟩
  have current : Real.Reads p (start (grid p) (n/128,b*p.heads+h,0) initial) μ := reads
  rw [Real.answer_real p _ μ current _ active,Real.expected_at p initial μ b h n d hh hd]

/-- Native interpretation uses the existing explicit scalar/fixed-ISA graph
and proved quotation execution. No InstructionAgreement or finite-word premise
is required. The finite byte result reuses the existing Tensor-memory binding. -/
theorem native_result (instructions : Native.Instructions)
    (p : Parameters) (storage : TensorStorage p) (initial : State Native.Word)
    (placement : Placement p storage) (flat : Fin placement.arena → BitVec 8)
    (binding : Native.WordBinding p storage placement flat initial)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel) p (grid p) work
        (Native.injectState initial)=some final ∧
      Native.WordResult instructions p initial (Native.projectState instructions p initial final) ∧
      Native.ByteResult instructions p storage placement initial flat
        (flattenBytes p storage placement (fun _=>Native.encode16) flat
          (Native.projectState instructions p initial final)) := by
  obtain ⟨final,run,values,frame⟩:=grid_result (fun _=>Native.algebra) (fun _=>Native.dotModel)
    p storage (Native.injectState initial) work legal
  have result : Native.WordResult instructions p initial (Native.projectState instructions p initial final) := by
    refine ⟨?_,?_⟩
    · intro b hb h hh n hn d hd
      change Native.projectCell instructions (Native.ownerAt p p.Out (logicalAddress p b h n d))
        (final.mem p.Out (logicalAddress p b h n d))=_
      rw [values b hb h hh n hn d hd]
      simp only [Native.ownerAt,if_pos rfl,Native.featureAt_output p storage b h n d hb hh hn hd]
      rfl
    · intro r off outside
      change Native.projectCell instructions (Native.ownerAt p r off) (final.mem r off)=_
      rw [frame r off outside]
      exact Native.injectState_project instructions _ initial r off
  exact ⟨final,run,result,Native.finite_result instructions p storage placement flat initial _ binding result⟩

#axiomsClean output_offset
#axiomsClean grid_result
#axiomsClean real_result
#axiomsClean native_result
#specNonCircular OutputBinding avoiding [runPrograms,FP.Structural.exec,Native.outputGraph]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Kernel
