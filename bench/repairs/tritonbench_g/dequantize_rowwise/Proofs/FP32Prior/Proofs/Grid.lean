import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Execution

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def atProgram (s : State α) (pid : Nat) : State α :=
  {s with pids:=fun ax=>if ax=0 then pid else 0}

def value (M : Algebra α) (p : Parameters) (s : State α) (r d : Nat) : α :=
  Execution.scalar M ((s.mem scaleRegion r).read .real) ((s.mem xRegion (r*p.cols+d)).read .real)

def execute (M : Algebra α) (p : Parameters) : List Nat → State α → Option (State α)
  | [],s=>some s
  | pid::tail,s=>(FP.Structural.exec M (kernel p) (atProgram s pid)).bind (execute M p tail)

theorem address_unique (cols r t d e : Nat) (hd : d<cols) (he : e<cols)
    (eq : r*cols+d=t*cols+e) : r=t ∧ d=e := by
  have positive : 0<cols := by omega
  have div := congrArg (fun x=>x/cols) eq
  have rows : r=t := by
    simpa [Nat.mul_comm r cols,Nat.mul_comm t cols,Nat.mul_add_div positive,
      Nat.div_eq_of_lt hd,Nat.div_eq_of_lt he] using div
  exact ⟨rows,by rw [rows] at eq;omega⟩

theorem at_correct (M : Algebra α) (p : Parameters) (s : State α) (pid : Nat) :
    ∃ t,FP.Structural.exec M (kernel p) (atProgram s pid)=some t ∧
      (∀ d,d<p.cols → t.mem outRegion (pid*p.cols+d)=.mk .fp16 (value M p s pid d)) ∧
      (∀ r o,(r≠outRegion ∨ ∀ d,d<p.cols → o≠pid*p.cols+d) → t.mem r o=s.mem r o) := by
  simpa only [atProgram,
    ↓reduceIte,Execution.cell,value] using Execution.complete_correct M p (atProgram s pid)

theorem order_correct (M : Algebra α) (p : Parameters) (order : List Nat) (s : State α) :
    ∃ t,execute M p order s=some t ∧
      (∀ r,r∈order → ∀ d,d<p.cols → t.mem outRegion (r*p.cols+d)=.mk .fp16 (value M p s r d)) ∧
      (∀ region o,(region≠outRegion ∨ ∀ r,r∈order → ∀ d,d<p.cols → o≠r*p.cols+d) → t.mem region o=s.mem region o) := by
  induction order generalizing s with
  | nil=>
    refine ⟨s,rfl,?_,?_⟩
    · intro r h; simp at h
    · intros; rfl
  | cons pid tail ih=>
    obtain ⟨u,hu,uread,uframe⟩:=at_correct M p s pid
    obtain ⟨t,ht,tread,tframe⟩:=ih u
    have same : ∀ r d,value M p u r d=value M p s r d := by
      intro r d
      simp only [value,uframe scaleRegion r (Or.inl (by decide)),
        uframe xRegion (r*p.cols+d) (Or.inl (by decide))]
    refine ⟨t,by simp [execute,hu,ht],?_,?_⟩
    · intro r hr d hd
      by_cases tailMember : r∈tail
      · rw [tread r tailMember d hd,same]
      · have equality : r=pid := by simpa [tailMember] using hr
        rw [tframe]
        · simpa only [equality] using uread d hd
        · right
          intro a ha e he eq
          have equal := address_unique p.cols r a d e hd he eq
          exact tailMember (equal.1 ▸ ha)
    · intro region o miss
      rw [tframe]
      · apply uframe
        exact miss.imp id (fun h d hd=>h pid (by simp) d hd)
      · exact miss.imp id (fun h r hr d hd=>h r (by simp [hr]) d hd)

def launch (M : Algebra α) (p : Parameters) (s : State α) := execute M p (List.range p.rows) s

theorem launch_correct (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t,launch M p s=some t ∧
      (∀ r,r<p.rows → ∀ d,d<p.cols → t.mem outRegion (r*p.cols+d)=.mk .fp16 (value M p s r d)) ∧
      (∀ region o,(region≠outRegion ∨ ∀ r,r<p.rows → ∀ d,d<p.cols → o≠r*p.cols+d) → t.mem region o=s.mem region o) := by
  simpa only [launch,List.mem_range] using order_correct M p (List.range p.rows) s

#axiomsClean order_correct
#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Grid
