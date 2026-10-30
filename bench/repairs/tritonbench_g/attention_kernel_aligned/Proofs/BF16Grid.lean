import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Output
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Whole
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned Source VeriTile.Bench.Repairs.AttentionKernelAligned.Full VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

abbrev Program := Nat×Nat×Nat
structure Grid where
  (x y z : Nat)

def programs (g : Grid) : List Program :=
  (List.range g.x).flatMap fun x => (List.range g.y).flatMap fun y =>
    (List.range g.z).map fun z => (x,y,z)

theorem mem_programs (g : Grid) (x y z : Nat) :
    (x,y,z)∈programs g ↔ x<g.x ∧ y<g.y ∧ z<g.z := by
  simp [programs,List.mem_flatMap,List.mem_map]

def ids (i : Program) : Nat → Nat :=
  fun ax => if ax=0 then i.1 else if ax=1 then i.2.1 else if ax=2 then i.2.2 else 0

def dimensions (g : Grid) : Nat → Nat :=
  fun ax => if ax=0 then g.x else if ax=1 then g.y else if ax=2 then g.z else 1

def start {α : Type} (g : Grid) (i : Program) (s : State α) : State α :=
  {s with regs:=fun _ _ _=>none,pids:=ids i,numPids:=dimensions g}

structure Separated (p : Parameters) : Prop where
  query : p.q≠p.output
  key : p.k≠p.output
  value : p.v≠p.output
  bias : p.bias≠p.output

variable {α : Type} [Inhabited α]

def ReadsEq (p : Parameters) (s t : State α) : Prop :=
  ∀ r, r≠p.output → s.mem r=t.mem r

theorem references_eq (P : Primitives α) (p : Parameters) (sep : Separated p)
    (s t : State α) (hp : s.pids=t.pids) (hr : ReadsEq p s t) :
    active p s=active p t ∧ address p s=address p t ∧ answer P p s=answer P p t := by
  have hq:=hr p.q sep.query
  have hk:=hr p.k sep.key
  have hv:=hr p.v sep.value
  have hb:=hr p.bias sep.bias
  have hu : advance P p s=advance P p t := by
    funext c a
    delta advance newMax alphaValue weights weightedDot score32 score rawScore qkDot
      qScaled scaleValue qValue kValue vValue b0Value b1Value
      qOffset kOffset vOffset b0Offset b1Offset row headBase kvBase valueBase biasBase
    simp only [hp,hq,hk,hv,hb]
  have hrnd : ∀ c,rounds P p s c=rounds P p t c := by
    intro c
    induction c with
    | zero=>rfl
    | succ c ih=>simp only [rounds,hu,ih]
  refine ⟨?_,?_,?_⟩
  · funext i
    simp only [active,row,hp]
  · funext i
    simp only [address,outputOffset,outputBase,row,hp]
  · funext i
    simp only [answer,outputValue,hrnd]

def Unique (p : Parameters) (g : Grid) (jobs : List Program) (s : State α) : Prop :=
  ∀ a∈jobs, ∀ b∈jobs, ∀ i j,
    active p (start g a s) i → active p (start g b s) j →
    address p (start g a s) i=address p (start g b s) j → a=b ∧ i=j

def Outside (p : Parameters) (g : Grid) (jobs : List Program) (s : State α)
    (r : RegionName) (o : Nat) : Prop :=
  r≠p.output ∨ ∀ a∈jobs, ∀ i, active p (start g a s) i → o≠address p (start g a s) i

def Post (M : Program → Algebra α) (dot : Program → Primitives α) (p : Parameters)
    (g : Grid) (jobs : List Program) (s t : State α) : Prop :=
  (∀ a∈jobs, ∀ i, active p (start g a s) i →
    t.mem p.output (address p (start g a s) i)=.mk .bf16 (answer (dot a) p (start g a s) i)) ∧
  (∀ r o, Outside p g jobs s r o → t.mem r o=s.mem r o)

def runPrograms (M : Program → Algebra α) (dot : Program → Primitives α) (p : Parameters) (g : Grid) :
    List Program → State α → Option (State α)
  | [], s => some s
  | a::jobs,s => (FP.Structural.exec (backend (M a) (dot a)) (VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source.source p) (start g a s)).bind
      (runPrograms M dot p g jobs)

/-- Source execution of all CTAs in any list order. Every CTA reads current
memory; output disjointness and the source frame establish initial-input readback. -/
theorem run_correct (M : Program → Algebra α) (dot : Program → Primitives α) (p : Parameters)
    (sep : Separated p) (hBN : 0<p.blockN) (g : Grid) (jobs : List Program) (s : State α) (unique : Unique p g jobs s) :
    ∃ t, runPrograms M dot p g jobs s=some t ∧ Post M dot p g jobs s t := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,by simp [Post,Outside]⟩
  | cons a jobs ih =>
    have hw : ∀ i j, active p (start g a s) i → active p (start g a s) j →
        address p (start g a s) i=address p (start g a s) j → i=j := by
      intro i j hi hj he
      exact (unique a (by simp) a (by simp) i j hi hj he).2
    obtain ⟨next,he,hm⟩:=source_memory (M a) (dot a) p (start g a s) hBN
    have hv : ∀ i, active p (start g a s) i → next.mem p.output (address p (start g a s) i)=
        .mk .bf16 (answer (dot a) p (start g a s) i) := by
      intro i hi
      rw [hm]
      exact readback _ _ _ i hi hw
    have hf : ∀ r o, (r≠p.output ∨ ∀ i, active p (start g a s) i → o≠address p (start g a s) i) →
        next.mem r o=s.mem r o := by
      intro r o ho
      rw [hm]
      exact frame _ _ _ r o ho
    have hr : ReadsEq p next s := by
      intro r hn
      funext o
      exact hf r o (Or.inl hn)
    have refs (b : Program) := references_eq (dot b) p sep
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
        have outside : Outside p g jobs next p.output (address p (start g a s) i) := by
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

#axiomsClean run_correct
#specNonCircular answer avoiding [runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Whole
