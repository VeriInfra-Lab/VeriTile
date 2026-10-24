import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealExecution
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealStorage
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

abbrev Heap := Nat → MemCell

def read (cell : MemCell) : ℝ := match cell.readAs .real with
  | some (some value) => value
  | _ => 0

def entry (h : Call) (p : Storage.Placement h) (row : Nat) (heap : Heap) : BlockState where
  mem := Storage.lift h p (MemCell.real 0) heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then row else 0
  numPids := fun ax => if ax=0 then h.B*h.H else 1
  undef := fun _ _ => 0

def execute (h : Call) (p : Storage.Placement h) (c : Config) (row : Nat) (heap : Heap) : Option Heap :=
  (exec (Kernels.source h c).eraseDType.toAlgKernel (entry h p row heap)).map
    (fun v => Storage.commit h p heap v.mem)

def input (h : Call) (p : Storage.Placement h) (heap : Heap) (row time : Nat) : ℝ :=
  read (heap (Storage.address p.backing.input (h.inputStrides.address h.H row time)))

def expected (h : Call) (p : Storage.Placement h) (_c : Config) (_positive : 0<_c.BT)
    (heap : Heap) (row time : Nat) : MemCell :=
  MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix (input h p heap row) h.T time)

theorem complete (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (row : Fin (h.B*h.H)) (heap : Heap) :
    ∃ final, execute h p c row.val heap=some final ∧
      (∀ off, off<p.backing.input.extent → final (Storage.address p.backing.input off)=heap (Storage.address p.backing.input off)) ∧
      (∀ time, time<h.T → final (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
        expected h p c positive heap row.val time) ∧
      (∀ addr, ¬Storage.writes h p [row] addr → final addr=heap addr) := by
  obtain ⟨v,he,hp⟩ := RealExecution.source_correct h layout c positive (entry h p row.val heap) row.isLt
  refine ⟨Storage.commit h p heap v.mem,?_,?_,?_,?_⟩
  · simp [execute,he]
  · exact fun off inside => Storage.commit_input h p heap v.mem off inside
  · intro time ht
    rw [Storage.commit_output h p heap v.mem _ (p.backing.outputWindow _ _ row.isLt ht)]
    erw [hp.1 time ht]
    simp [expected,entry,BlockState.readMem,input,read,Storage.lift_input]
    rfl
  · intro addr outside
    unfold Storage.commit
    cases eq : Storage.decode p.backing.output addr with
    | none => rfl
    | some off =>
      have spec := Storage.decode_some _ _ _ eq
      have no : ¬RealMemory.footprint h row.val (h.output,off) := by
        rintro ⟨_,time,ht,eq⟩
        change off=h.outputStrides.address h.H row.val time at eq
        exact outside ⟨row,by simp,time,ht,by rw [eq] at spec; exact spec.2⟩
      change v.mem h.output off=heap addr
      rw [←hp.2 h.output off no]
      change Storage.lift h p (MemCell.real 0) heap h.output off=heap addr
      rw [Storage.lift_output h layout p,←spec.2]

#print axioms complete
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealStorage
