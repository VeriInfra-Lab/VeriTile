import bench.tritonbench_g.attention_forward_triton.Proofs.HoistedOutput
import bench.tritonbench_g.attention_forward_triton.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedWhole
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def runPrograms (M : Program → Algebra α) (dot : Program → Dot α) (p : Parameters) (g : Whole.Grid) :
    List Program → State α → Option (State α)
  | [], s => some s
  | a::jobs,s => (FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (withDot (M a) (dot a))) (HoistedSource.kernel p) (start g a s)).bind
      (runPrograms M dot p g jobs)

/-- Source execution of all CTAs in any list order. Every CTA reads current
memory; output disjointness and the source frame establish initial-input readback. -/
theorem run_correct (M : Program → Algebra α) (dot : Program → Dot α) (p : Parameters)
    (sep : Separated p) (g : Whole.Grid) (jobs : List Program) (s : State α) (unique : Unique p g jobs s) :
    ∃ t, runPrograms M dot p g jobs s=some t ∧ Post M dot p g jobs s t := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,by simp [Post,Outside]⟩
  | cons a jobs ih =>
    have hw : ∀ i j, active p (start g a s) i → active p (start g a s) j →
        address p (start g a s) i=address p (start g a s) j → i=j := by
      intro i j hi hj he
      exact (unique a (by simp) a (by simp) i j hi hj he).2
    obtain ⟨next,he,hm⟩:=HoistedSource.source_memory (M a) (dot a) p (start g a s)
    have hv : ∀ i, active p (start g a s) i → next.mem p.Out (address p (start g a s) i)=
        .mk .fp16 (answer (M a) (dot a) p (start g a s) i) := by
      intro i hi
      rw [hm]
      exact readback _ _ _ _ i hi hw
    have hf : ∀ r o, (r≠p.Out ∨ ∀ i, active p (start g a s) i → o≠address p (start g a s) i) →
        next.mem r o=s.mem r o := by
      intro r o ho
      rw [hm]
      exact frame _ _ _ _ r o ho
    have hr : ReadsEq p next s := by
      intro r hn
      funext o
      exact hf r o (Or.inl hn)
    have refs (b : Program) := references_eq (M b) (dot b) p sep
      (start g b next) (start g b s) rfl hr
    have uniqNext : Unique p g jobs next := by
      intro b hb c hc i j hi hj hij
      rw [(refs b).1] at hi
      rw [(refs c).1] at hj
      rw [(refs b).2.1,(refs c).2.1] at hij
      exact unique b (by simp [hb]) c (by simp [hc]) i j hi hj hij
    obtain ⟨t,ht,htv,htf⟩:=ih next uniqNext
    refine ⟨t,?_,?_,?_⟩
    · simp only [runPrograms,he,Option.bind_some]
      exact ht
    · intro b hb i hi
      by_cases hbJobs : b∈jobs
      · have h:=htv b hbJobs i (by rw [(refs b).1];exact hi)
        simpa only [(refs b).2.1,(refs b).2.2] using h
      · have hab : b=a := by simpa [hbJobs] using hb
        subst b
        have outside : Outside p g jobs next p.Out (address p (start g a s) i) := by
          right
          intro b hb j hj heq
          rw [(refs b).1] at hj
          rw [(refs b).2.1] at heq
          have same := (unique a (by simp) b (by simp [hb]) i j hi hj heq).1
          exact hbJobs (same ▸ hb)
        rw [htf _ _ outside]
        exact hv i hi
    · intro r o ho
      have outTail : Outside p g jobs next r o := by
        rcases ho with ho|ho
        · exact Or.inl ho
        · right
          intro b hb i hi
          rw [(refs b).1] at hi
          rw [(refs b).2.1]
          exact ho b (by simp [hb]) i hi
      rw [htf r o outTail]
      apply hf
      rcases ho with ho|ho
      · exact Or.inl ho
      · exact Or.inr (fun i hi=>ho a (by simp) i hi)

theorem grid_result {α : Type} [Inhabited α]
    (M : Program → Algebra α) (dot : Program → Dot α)
    (p : Parameters) (sep : Separated p) (injective : RawKernel.OutputInjective p) (initial : State α)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms M dot p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16
          (answer (M (n/128,b*p.heads+h,0)) (dot (n/128,b*p.heads+h,0)) p
            (start (grid p) (n/128,b*p.heads+h,0) initial) (lane n d hd))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off) := by
  have unique : Unique p (grid p) work initial := by
    intro a ha b hb i j hi hj he
    exact RawKernel.grid_unique p injective initial a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
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


#axiomsClean run_correct
#axiomsClean grid_result
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedWhole
