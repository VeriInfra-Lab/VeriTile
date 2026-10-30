import bench.tritonbench_g.bmm_optimized.Proofs.Complete
import bench.tritonbench_g.bmm_optimized.Proofs.Writes
import bench.tritonbench_g.bmm_optimized.Proofs.Ownership

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
open VeriTile.Triton FP.Structural
open BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem scatter_commit (o : Scalar α) (s : Shape) (c : Config) (initial current : State α) :
    scatter o (parameters s c) initial current=
      commitState (sourceWrites o (parameters s c) initial) current := by
  unfold scatter sourceWrites commitState
  generalize TileShape.allIndices [(parameters s c).TM,(parameters s c).TN]=lanes
  induction lanes generalizing current with
  | nil=>rfl
  | cons i lanes ih=>
    by_cases ha : Execution.active (parameters s c) initial i
    · simp only [List.foldl_cons,List.filterMap_cons,ha,ite_true]
      exact ih _
    · simp only [List.foldl_cons,List.filterMap_cons,ha,ite_false]
      exact ih current

def Frame (s : Shape) (initial final : State α) : Prop :=
  ∀ (r : RegionName) (off : Nat),r≠"O" ∨ s.size≤off → final.mem r off=initial.mem r off

theorem programExecution (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α)
    (pid : Program) (hp : pid∈programs s c) (m : Memory α) (state : State α)
    (related : Related m state) :
    ∃ final,exec o rest (parameters s c) (start s c pid state)=some final ∧
      Related (sourceProgram s c o pid m) final ∧ Frame s state final := by
  obtain ⟨final,run,eq⟩:=correctness o rest (parameters s c) (start s c pid state)
  rw [scatter_commit,writes_equal s c o pid hp m state related] at eq
  have relatedStart : Related m (start s c pid state) :=
    ⟨related.a,related.b,related.output⟩
  have result:=related_commit (programWrites s c o pid m) m _ relatedStart
  refine ⟨final,run,?_,?_⟩
  · constructor
    · intro off;rw [eq];exact result.a off
    · intro off;rw [eq];exact result.b off
    · intro off;rw [eq];exact result.output off
  · intro r off outside
    rw [eq,commit_frame]
    · rfl
    · intro w hw
      rcases outside with other|bound
      · exact Or.inl other
      · obtain ⟨lane,hl,x,hx,active,eq⟩:=(programWrites_member s c o pid m w).mp hw
        have upper:=address_bound s x active
        subst w
        exact Or.inr (by simp only;omega)

/-- Each CTA is interpreted from the memory left by preceding CTAs. -/
def grid (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α) :
    List Program → State α → Option (State α)
  | [],state=>some state
  | pid::ps,state=>do
      let next←exec o rest (parameters s c) (start s c pid state)
      grid s c o rest ps next

theorem gridExecution (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α)
    (ps : List Program) (legal : ∀ pid∈ps,pid∈programs s c)
    (m : Memory α) (state : State α) (related : Related m state) :
    ∃ final,grid s c o rest ps state=some final ∧
      Related (sourcePrograms s c o ps m) final ∧ Frame s state final := by
  induction ps generalizing m state with
  | nil=>exact ⟨state,rfl,related,fun _ _ _=>rfl⟩
  | cons pid ps ih=>
    obtain ⟨next,hr,hn,hf⟩:=programExecution s c o rest pid (legal pid (by simp)) m state related
    obtain ⟨final,run,rel,frame⟩:=ih (fun q hq=>legal q (by simp [hq])) _ next hn
    refine ⟨final,by simp [grid,hr,bind,Option.bind,run],rel,?_⟩
    intro r off outside
    exact (frame r off outside).trans (hf r off outside)

theorem orderedExecution (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α)
    (ps : List Program) (order : ps.Perm (programs s c))
    (m : Memory α) (state : State α) (related : Related m state) :
    ∃ final,grid s c o rest ps state=some final ∧
      (∀ x,s.Active x → final.mem "O" (s.o x.batch x.row x.column)=
        .mk .real (ordered s o m x)) ∧ Frame s state final := by
  obtain ⟨final,run,rel,frame⟩:=gridExecution s c o rest ps
    (fun _ h=>order.mem_iff.mp h) m state related
  refine ⟨final,run,?_,frame⟩
  intro x hx
  rw [rel.output,any_program_order s c o ps order m x hx]

#axiomsClean programExecution
#axiomsClean gridExecution
#axiomsClean orderedExecution
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
