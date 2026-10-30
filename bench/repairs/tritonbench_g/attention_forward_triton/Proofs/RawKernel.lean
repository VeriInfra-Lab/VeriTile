import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RealCorrect
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.SourceFPEquiv
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.NativeReferenceBinding

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.RawKernel
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

/-- Raw affine output addresses distinguish all active logical coordinates. -/
def OutputInjective (p : Parameters) : Prop :=
  ∀ b<p.batches, ∀ h<p.heads, ∀ n<p.length, ∀ d<96,
    ∀ b'<p.batches, ∀ h'<p.heads, ∀ n'<p.length, ∀ d'<96,
      logicalAddress p b h n d=logicalAddress p b' h' n' d' →
        b=b' ∧ h=h' ∧ n=n' ∧ d=d'

theorem grid_unique (p : Parameters) (injective : OutputInjective p) (s : State α) :
    Unique p (grid p) (programs (grid p)) s := by
  intro a ha b hb i j hi hj he
  rcases a with ⟨ax,ay,az⟩
  rcases b with ⟨bx,byy,bz⟩
  obtain ⟨hax,hay,haz⟩:=(mem_programs (grid p) ax ay az).mp ha
  obtain ⟨hbx,hby,hbz⟩:=(mem_programs (grid p) bx byy bz).mp hb
  have ha0 : az=0 := by change az<1 at haz;omega
  have hb0 : bz=0 := by change bz<1 at hbz;omega
  obtain ⟨hH,haB,haH⟩:=head_decode p ay hay
  obtain ⟨_,hbB,hbH⟩:=head_decode p byy hby
  change ax*128+i.1.val<p.length ∧ i.2.1.val<96 at hi
  change bx*128+j.1.val<p.length ∧ j.2.1.val<96 at hj
  change ay/p.heads*p.o.batch+ay%p.heads*p.o.head+(ax*128+i.1.val)*p.o.row+i.2.1.val*p.o.dim=
    byy/p.heads*p.o.batch+byy%p.heads*p.o.head+(bx*128+j.1.val)*p.o.row+j.2.1.val*p.o.dim at he
  obtain ⟨hz,hh,hn,hd⟩:=injective _ haB _ haH _ hi.1 _ hi.2
    _ hbB _ hbH _ hj.1 _ hj.2 he
  have hy : ay=byy := by
    have h1:=Nat.div_add_mod ay p.heads
    have h2:=Nat.div_add_mod byy p.heads
    rw [hz,hh] at h1
    omega
  have hx : ax=bx := by have:=i.1.isLt;have:=j.1.isLt;omega
  have hr : i.1.val=j.1.val := by omega
  refine ⟨by simp [hx,hy,ha0,hb0],?_⟩
  rcases i with ⟨ir,id,⟨⟩⟩
  rcases j with ⟨jr,jd,⟨⟩⟩
  simp only [Prod.mk.injEq,Fin.mk.injEq,true_and,and_true]
  exact ⟨Fin.ext hr,Fin.ext hd⟩

open Native in
theorem featureAt_output (p : Parameters) (injective : OutputInjective p) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<96) :
    featureAt p (logicalAddress p b h n d)=d := by
  have he : ∃ e,Feature p (logicalAddress p b h n d) e := ⟨d,hd,b,hb,h,hh,n,hn,rfl⟩
  rw [featureAt,dif_pos he]
  obtain ⟨hed,bb,hbb,hh',hhh,nn,hnn,heq⟩:=Classical.choose_spec he
  exact (injective b hb h hh n hn d hd bb hbb hh' hhh nn hnn _ hed heq).2.2.2.symm

theorem grid_result {α : Type} [Inhabited α]
    (M : Program → Algebra α) (dot : Program → Dot α)
    (p : Parameters) (sep : Separated p) (injective : OutputInjective p) (initial : State α)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms M dot p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16
          (answer (M (n/128,b*p.heads+h,0)) (dot (n/128,b*p.heads+h,0)) p
            (start (grid p) (n/128,b*p.heads+h,0) initial) (lane n d hd))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off) := by
  have unique : Unique p (grid p) work initial := by
    intro a ha b hb i j hi hj he
    exact grid_unique p injective initial a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
  obtain ⟨final,hr,hv,hf⟩:=run_correct M dot p sep (grid p) work initial unique
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

theorem real_result (p : Parameters) (sep : Separated p) (injective : OutputInjective p) (initial : State Real.R)
    (μ : RegionName → Nat → ℝ) (reads : Real.Reads p initial μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off) := by
  obtain ⟨final,run,values,frame⟩:=grid_result (fun _=>Real.algebra) (fun _=>Real.dotModel) p sep injective initial work legal
  refine ⟨final,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd]
  have active : Source.active p (start (grid p) (n/128,b*p.heads+h,0) initial) (lane n d hd) :=
    ⟨by change n/128*128+n%128<p.length;omega,hd⟩
  have current : Real.Reads p (start (grid p) (n/128,b*p.heads+h,0) initial) μ := reads
  rw [Real.answer_real p _ μ current _ active,Real.expected_at p initial μ b h n d hh hd]

theorem native_result (instructions : Native.Instructions)
    (p : Parameters) (sep : Separated p) (injective : OutputInjective p) (initial : State Native.Word)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel) p (grid p) work
        (Native.injectState initial)=some final ∧
      Native.WordResult instructions p initial (Native.projectState instructions p initial final) := by
  obtain ⟨final,run,values,frame⟩:=grid_result (fun _=>Native.algebra) (fun _=>Native.dotModel)
    p sep injective (Native.injectState initial) work legal
  have result : Native.WordResult instructions p initial (Native.projectState instructions p initial final) := by
    refine ⟨?_,?_⟩
    · intro b hb h hh n hn d hd
      change Native.projectCell instructions (Native.ownerAt p p.Out (logicalAddress p b h n d))
        (final.mem p.Out (logicalAddress p b h n d))=_
      rw [values b hb h hh n hn d hd]
      simp only [Native.ownerAt,if_pos rfl,featureAt_output p injective b h n d hb hh hn hd]
      rfl
    · intro r off outside
      change Native.projectCell instructions (Native.ownerAt p r off) (final.mem r off)=_
      rw [frame r off outside]
      exact Native.injectState_project instructions _ initial r off
  exact ⟨final,run,result⟩

#axiomsClean grid_result
#axiomsClean real_result
#axiomsClean native_result
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.RawKernel
