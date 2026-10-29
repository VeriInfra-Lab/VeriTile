import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealExecution
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Stream

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealStorage
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
abbrev Heap := Nat → MemCell

def read (cell : MemCell) : ℝ := match cell.readAs .real with
  | some (some value) => value
  | _ => 0

def entry (h : Call) (p : Storage.Placement h) (prog : Geometry.Program h)
    (padding : RegionName → Nat → ℝ) (heap : Heap) : BlockState where
  mem := Storage.lift h p (MemCell.real 0) heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then prog.feature else if ax=1 then prog.row else 0
  numPids := fun ax => if ax=0 then Geometry.features h else if ax=1 then h.B*h.H else 1
  undef := padding

def execute (h : Call) (p : Storage.Placement h) (padding : Config → Geometry.Program h → RegionName → Nat → ℝ)
    (c : Config) (prog : Geometry.Program h) (heap : Heap) : Option Heap :=
  (exec (Kernels.source h c).eraseDType.toAlgKernel (entry h p prog (padding c prog) heap)).map
    (fun v => Storage.commit h p heap v.mem)

def expected (h : Call) (p : Storage.Placement h) (_c : Config) (heap : Heap)
    (prog : Geometry.Program h) (time : Nat) (j : Fin BS) : ℝ :=
  RealData.expected h prog.feature prog.row
    (fun off => read (heap (Storage.address p.backing.input off))) time j

theorem complete (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (padding : Config → Geometry.Program h → RegionName → Nat → ℝ)
    (c : Config) (chosen : c∈configurations) (prog : Geometry.Program h) (heap : Heap) :
    ∃ final, execute h p padding c prog heap=some final ∧ Stream.StableInput h p heap final ∧
      (∀ time j, Grid.active h prog time j →
        read (final (Storage.address p.backing.output (Grid.offset h prog time j)))=expected h p c heap prog time j) ∧
      (∀ addr, ¬Storage.writes h p [prog] addr → final addr=heap addr) := by
  obtain ⟨v,he,out,frame⟩ := RealExecution.source_executes h c chosen prog layout
    (entry h p prog (padding c prog) heap) (by simp [entry]) (by simp [entry])
  refine ⟨Storage.commit h p heap v.mem,?_,?_,?_,?_⟩
  · simp [execute,he]
  · exact fun off inside => Storage.commit_input h p heap v.mem off inside
  · intro time j ha
    dsimp only [Grid.offset]
    rw [Storage.commit_output h p heap v.mem _ (p.backing.outputWindow _ _ _ prog.rowInside ha.1 ha.2)]
    have output := out time j ha
    change read (v.mem h.output (h.outputStrides.address h.H prog.row time (prog.feature*BS+j.val)))=_ at output
    rw [output]
    unfold expected RealData.expected
    simp only [entry,BlockState.readMem,Storage.lift_input,read]
    rfl
  · exact Storage.commit_frame h layout p (MemCell.real 0) heap v.mem prog frame

theorem expected_stable (h : Call) (p : Storage.Placement h) (c : Config)
    (a b : Heap) (stable : Stream.StableInput h p a b) (prog : Geometry.Program h)
    (time : Nat) (j : Fin BS) (active : Grid.active h prog time j) : expected h p c b prog time j=expected h p c a prog time j := by
  unfold expected RealData.expected
  apply Finset.sum_congr rfl
  intro k hk
  have kt := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
  dsimp only
  rw [stable _ (p.backing.inputWindow _ _ _ prog.rowInside kt active.2)]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealStorage
