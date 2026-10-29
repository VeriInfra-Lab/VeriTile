import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPExecution
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Stream

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPStorage
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]
abbrev Heap (α : Type) := Nat → Cell α
abbrev Padding (α : Type) := (dtype : TileDType) → RegionName → Nat → Value α dtype

def entry (h : Call) (p : Storage.Placement h) (prog : Geometry.Program h)
    (padding : Padding α) (heap : Heap α) : State α where
  mem := Storage.lift h p (.mk .real default) heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then prog.feature else if ax=1 then prog.row else 0
  numPids := fun ax => if ax=0 then Geometry.features h else if ax=1 then h.B*h.H else 1
  undef := padding

def execute (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (padding : Config → Geometry.Program h → Padding α)
    (c : Config) (prog : Geometry.Program h) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (machine M dot) (Kernels.source h c) (entry h p prog (padding c prog) heap)).map
    (fun v => Storage.commit h p heap v.mem)

def expected (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (padding : Config → Geometry.Program h → Padding α)
    (c : Config) (heap : Heap α) (prog : Geometry.Program h) (time : Nat) (j : Fin BS) : Cell α :=
  if hp : 0<c.BT then FPMemory.stored M h
    (FPExecution.expected M dot h c hp prog (entry h p prog (padding c prog) heap) time j)
  else .mk .real default

theorem complete (M : Algebra α) (dot : DotBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (padding : Config → Geometry.Program h → Padding α)
    (c : Config) (chosen : c∈configurations) (prog : Geometry.Program h) (heap : Heap α) :
    ∃ final, execute M dot h p padding c prog heap=some final ∧ Stream.StableInput h p heap final ∧
      (∀ time j, Grid.active h prog time j →
        final (Storage.address p.backing.output (Grid.offset h prog time j))=expected M dot h p padding c heap prog time j) ∧
      (∀ addr, ¬Storage.writes h p [prog] addr → final addr=heap addr) := by
  have positive := (Kernels.configurations_positive c chosen).1
  obtain ⟨v,he,out,frame⟩ := FPExecution.source_correct M dot h layout c positive prog
    (entry h p prog (padding c prog) heap) (by simp [entry]) (by simp [entry])
  refine ⟨Storage.commit h p heap v.mem,?_,?_,?_,?_⟩
  · simp [execute,he]
  · exact fun off inside => Storage.commit_input h p heap v.mem off inside
  · intro time j ha
    dsimp only [Grid.offset]
    rw [Storage.commit_output h p heap v.mem _ (p.backing.outputWindow _ _ _ prog.rowInside ha.1 ha.2)]
    exact (out time j ha.1 ha.2).trans (by simp only [expected,dif_pos positive])
  · apply Storage.commit_frame h layout p (.mk .real default) heap v.mem prog
    intro r off hn
    apply frame r off
    intro hf
    rcases hf with ⟨hr,t,j,ht,hj,eq⟩
    exact hn ⟨hr,t,j,⟨ht,hj⟩,eq⟩

theorem loaded_stable (M : Algebra α) (h : Call) (p : Storage.Placement h)
    (c : Config) (prog : Geometry.Program h) (padding : Padding α)
    (a b : Heap α) (stable : Stream.StableInput h p a b) (q : Nat) :
    loaded M h c prog.feature prog.row (entry h p prog padding b) q=
      loaded M h c prog.feature prog.row (entry h p prog padding a) q := by
  funext i
  unfold loaded
  by_cases ha : Geometry.active h c prog.feature q i.1.val i.2.1.val
  · have ht : Geometry.time h c q i.1.val<h.T := by
      have hi := ha.1
      dsimp [Geometry.activeTime] at hi
      simp only [Geometry.time,if_pos ha.1]
      omega
    have hj : Geometry.column h prog.feature i.2.1.val<h.S := by
      have hi := ha.2
      dsimp [Geometry.activeColumn] at hi
      simp only [Geometry.column,if_pos ha.2]
      omega
    simp only [if_pos ha,entry,Storage.lift_input,stable _ (p.backing.inputWindow _ _ _ prog.rowInside ht hj)]
  · simp only [if_neg ha]

theorem carry_stable (M : Algebra α) (h : Call) (p : Storage.Placement h)
    (c : Config) (prog : Geometry.Program h) (padding : Padding α)
    (a b : Heap α) (stable : Stream.StableInput h p a b) (q : Nat) :
    FPExecution.carry M h c prog.feature prog.row (entry h p prog padding b) q=
      FPExecution.carry M h c prog.feature prog.row (entry h p prog padding a) q := by
  induction q with
  | zero => rfl
  | succ q ih =>
    simp only [FPExecution.carry,loaded_stable M h p c prog padding a b stable q,ih]

theorem expected_stable (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (padding : Config → Geometry.Program h → Padding α) (c : Config)
    (a b : Heap α) (stable : Stream.StableInput h p a b) (prog : Geometry.Program h)
    (time : Nat) (j : Fin BS) :
    expected M dot h p padding c b prog time j=expected M dot h p padding c a prog time j := by
  unfold expected
  split
  · unfold FPExecution.expected
    rw [loaded_stable M h p c prog (padding c prog) a b stable,
      carry_stable M h p c prog (padding c prog) a b stable]
  · rfl

#axiomsClean complete
#axiomsClean expected_stable
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPStorage
