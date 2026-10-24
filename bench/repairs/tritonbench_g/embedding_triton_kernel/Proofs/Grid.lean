import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Execution
import bench.repairs.tritonbench_g.embedding_triton_kernel.Contract

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Grid
open VeriTile.Triton FP.Structural
open Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def atProgram (s : State α) (pid : Nat) : State α :=
  {s with pids:=fun ax=>if ax=0 then pid else 0}

def expected (zero : α) (p : Parameters) (s : State α) (r d : Nat) : α :=
  target zero p.first p.last (fun r=>(s.mem p.ids r).read .int)
    (fun r d=>(s.mem p.weight (r*p.weightStride+d)).read .real) r d

def execute (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters) :
    List Nat → State α → Option (State α)
  | [],s=>some s
  | pid::tail,s=>(FP.Structural.exec (backend rest zero) (source index p)
      (atProgram s pid)).bind (execute rest zero index p tail)

theorem at_correct (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (layout : Layout p) (s : State α) (pid : Nat) :
    ∃ t,FP.Structural.exec (backend rest zero) (source index p) (atProgram s pid)=some t ∧
      (∀ r,r<p.n → r/64=pid → ∀ d,d<p.hidden →
        t.mem p.output (r*p.outputStride+d)=.mk .real (expected zero p s r d)) ∧
      (∀ region o,(region≠p.output ∨ ∀ r,r<p.n → r/64=pid →
        ∀ d,d<p.hidden → o≠r*p.outputStride+d) → t.mem region o=s.mem region o) := by
  obtain ⟨t,run,_,readback,frame⟩ := program_correct rest zero index p layout (atProgram s pid)
  refine ⟨t,run,?_,?_⟩
  · intro r hr hp d hd
    have eq : row (atProgram s pid) (r%64)=r := by
      simp only [row,atProgram,↓reduceIte]
      omega
    have h := readback (r%64) (Nat.mod_lt _ (by omega)) (by simpa [eq] using hr) d hd
    have values : value zero p (atProgram s pid) (r%64) d=expected zero p s r d := by
      simp only [value,valid,token,eq]
      simp [expected,target,atProgram,hr]
    simpa only [eq,values] using h
  · intro region o miss
    apply frame
    rcases miss with h|h
    · exact Or.inl h
    · right
      intro a ha hr d hd
      apply h (row (atProgram s pid) a) hr
      · change (pid*64+a)/64=pid
        omega
      · exact hd

theorem expected_stable (zero : α) (p : Parameters) (s t : State α)
    (hi : ∀ o,t.mem p.ids o=s.mem p.ids o)
    (hw : ∀ o,t.mem p.weight o=s.mem p.weight o) (r d : Nat) :
    expected zero p t r d=expected zero p s r d := by
  simp only [expected,target,hi,hw]

theorem order_correct (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (layout : Layout p) (order : List Nat) (s : State α) :
    ∃ t,execute rest zero index p order s=some t ∧
      (∀ r,r<p.n → r/64∈order → ∀ d,d<p.hidden →
        t.mem p.output (r*p.outputStride+d)=.mk .real (expected zero p s r d)) ∧
      (∀ region o,(region≠p.output ∨ ∀ r,r<p.n → r/64∈order →
        ∀ d,d<p.hidden → o≠r*p.outputStride+d) → t.mem region o=s.mem region o) := by
  induction order generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro r _ hm; simp at hm
    · intros; rfl
  | cons pid tail ih =>
    obtain ⟨u,hu,uread,uframe⟩ := at_correct rest zero index p layout s pid
    obtain ⟨t,ht,tread,tframe⟩ := ih u
    have same : ∀ r d,expected zero p u r d=expected zero p s r d :=
      expected_stable zero p s u
        (fun o=>uframe _ _ (Or.inl layout.ids_distinct))
        (fun o=>uframe _ _ (Or.inl layout.weight_distinct))
    refine ⟨t,by simp [execute,hu,ht],?_,?_⟩
    · intro r hr hm d hd
      by_cases htail : r/64∈tail
      · rw [tread r hr htail d hd,same]
      · have he : r/64=pid := by simpa [htail] using hm
        rw [tframe]
        · exact uread r hr he d hd
        · right
          intro a ha ham e heq eq
          have h := address_unique p layout r a d e hd heq eq
          exact htail (h.1 ▸ ham)
    · intro region o miss
      rw [tframe]
      · apply uframe
        exact miss.imp id (fun h r hr hp d hd=>h r hr (by simp [hp]) d hd)
      · exact miss.imp id (fun h r hr hp d hd=>h r hr (by simp [hp]) d hd)

theorem row_covered (n r : Nat) (hr : r<n) : r/64<grid n := by
  unfold grid
  omega

def launch (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters) (s : State α) :=
  execute rest zero index p (List.range (grid p.n)) s

/-- Every actual program in the host grid executes the full typed repaired
source. Every active coordinate is read back from the initial inputs; the
frame includes padding, extra output coordinates and all other regions. -/
theorem launch_correct (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (layout : Layout p) (s : State α) :
    ∃ t,launch rest zero index p s=some t ∧
      (∀ r,r<p.n → ∀ d,d<p.hidden →
        t.mem p.output (r*p.outputStride+d)=.mk .real (expected zero p s r d)) ∧
      (∀ region o,(region≠p.output ∨ ∀ r,r<p.n →
        ∀ d,d<p.hidden → o≠r*p.outputStride+d) → t.mem region o=s.mem region o) := by
  obtain ⟨t,run,readback,frame⟩ := order_correct rest zero index p layout (List.range (grid p.n)) s
  refine ⟨t,run,?_,?_⟩
  · intro r hr d hd
    exact readback r hr (by simpa using row_covered p.n r hr) d hd
  · intro region o miss
    apply frame
    exact miss.imp id (fun h r hr _ d hd=>h r hr d hd)

#axiomsClean at_correct
#axiomsClean order_correct
#axiomsClean row_covered
#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Grid
