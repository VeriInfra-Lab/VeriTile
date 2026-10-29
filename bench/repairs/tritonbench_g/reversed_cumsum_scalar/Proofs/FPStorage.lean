import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPReference
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPStorage
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

abbrev Heap (α : Type) := Nat → Cell α

def entry (h : Call) (p : Storage.Placement h) (row : Nat) (heap : Heap α) : State α where
  mem := Storage.lift h p (.mk .real default) heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then row else 0
  numPids := fun ax => if ax=0 then h.B*h.H else 1
  undef := fun d _ _ => defaultValue d

/-- Interpret the actual typed source through physical byte-addressed storage.
Storage.commit_write proves the individual store adapter, including holes. -/
def execute (M : Algebra α) (scan : ScanBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (row : Nat) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (machine M scan) (Kernels.source h c) (entry h p row heap)).map
    (fun v => Storage.commit h p heap v.mem)

def input (h : Call) (p : Storage.Placement h) (heap : Heap α) (row time : Nat) : α :=
  toFloat h.inputDType ((heap (Storage.address p.backing.input (h.inputStrides.address h.H row time))).read h.inputDType.toTileDType)

def expected (M : Algebra α) (scan : ScanBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (heap : Heap α) (row time : Nat) : Cell α :=
  FPMemory.stored M h (FPExecution.expected M scan h c positive (input h p heap row) time)

theorem complete (M : Algebra α) (scan : ScanBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (c : Config) (positive : 0<c.BT) (row : Fin (h.B*h.H)) (heap : Heap α) :
    ∃ final, execute M scan h p c row.val heap=some final ∧
      (∀ off, off<p.backing.input.extent → final (Storage.address p.backing.input off)=heap (Storage.address p.backing.input off)) ∧
      (∀ time, time<h.T → final (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
        expected M scan h p c positive heap row.val time) ∧
      (∀ addr, ¬Storage.writes h p [row] addr → final addr=heap addr) := by
  obtain ⟨v,he,hp⟩ := FPExecution.source_correct M scan h layout c positive (entry h p row.val heap) row.isLt
  refine ⟨Storage.commit h p heap v.mem,?_,?_,?_,?_⟩
  · simp [execute,he]
  · exact fun off inside => Storage.commit_input h p heap v.mem off inside
  · intro time ht
    rw [Storage.commit_output h p heap v.mem _ (p.backing.outputWindow _ _ row.isLt ht)]
    erw [hp.1 time ht]
    congr 1
    apply FPReference.expected_congr
    intro t ht
    simp [FPExecution.input,entry,Storage.lift_input,input]
  · intro addr outside
    unfold Storage.commit
    cases eq : Storage.decode p.backing.output addr with
    | none => rfl
    | some off =>
      have spec := Storage.decode_some _ _ _ eq
      have no : ¬FPMemory.footprint h row.val (h.output,off) := by
        rintro ⟨_,time,ht,eq⟩
        change off=h.outputStrides.address h.H row.val time at eq
        exact outside ⟨row,by simp,time,ht,by rw [eq] at spec; exact spec.2⟩
      change v.mem h.output off=heap addr
      rw [←hp.2 h.output off no]
      change Storage.lift h p (.mk .real default) heap h.output off=heap addr
      rw [Storage.lift_output h layout p,←spec.2]

#print axioms complete
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPStorage
