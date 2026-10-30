import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOOperands
import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOSchedule

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem patch_readonly (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (current : (IO.programIO x t o rest lanes).RawOutputs) (base : RegisterState (SourceGraph.Node α))
    (r : RegionName) (offset : Nat) (notOut : r≠"out") :
    ((IO.programIO x t o rest lanes).patchOutputs v current base).mem r offset=base.mem r offset := by
  apply (IO.programIO x t o rest lanes).toMemoryIO.patchOutputs_frame v current base
  constructor
  · intro port j _
    have hz : port=0 := Fin.eq_zero port
    subst port
    exact Or.inl notOut
  · intro i
    exact Fin.elim0 i

theorem operand_binding (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (domain : IO.domain x.shape o v) (base : RegisterState (SourceGraph.Node α))
    (bound : (IO.programIO x t o rest lanes).InputCells v base)
    (current : (IO.programIO x t o rest lanes).RawOutputs) (p : IO.Program) :
    OperandBinding x.shape o v (SourceGraph.observeState o
      ((IO.invocation x t o rest p).initial ((IO.programIO x t o rest lanes).patchOutputs v current base))) := by
  have read (r : RegionName) (i : Nat) (ne : r≠"out") :
      (SourceGraph.observeState o ((IO.invocation x t o rest p).initial
        ((IO.programIO x t o rest lanes).patchOutputs v current base))).mem r i=
        mapCell (SourceGraph.observe o) (base.mem r i) := by
    change mapCell _ (((IO.invocation x t o rest p).initial _).mem r i)=_
    rw [Invocation.initial_mem,patch_readonly x t o rest lanes v current base r i ne]
  constructor
  · intro i hi
    rw [read "a" i (by decide)]
    have h := bound 0 (⟨i,hi⟩,()) (by trivial)
    change base.mem "a" i=Cell.mk .real (.value (v 0 (⟨i,hi⟩,()))) at h
    rw [h]
    simp only [mapCell,mapValue,Cell.read,SourceGraph.observe,operands,dif_pos hi,dite_true]
  · intro i hi
    rw [read "scale" i (by decide)]
    have h := bound 2 (⟨i,hi⟩,()) (by trivial)
    change base.mem "scale" i=Cell.mk .real (.value (v 2 (⟨i,hi⟩,()))) at h
    rw [h]
    simp only [mapCell,mapValue,Cell.read,SourceGraph.observe,operands,dif_pos hi,dite_true]
  · intro i hi
    rw [read "weight" i (by decide)]
    have h := bound 1 (⟨i,hi⟩,()) (by trivial)
    change base.mem "weight" i=Cell.mk (α := SourceGraph.Node α) .int (v 1 (⟨i,hi⟩,())) at h
    rw [h]
    simpa only [mapCell,mapValue,Cell.read,operands,dif_pos hi,dite_true] using (domain.1 ⟨i,hi⟩).symm
  · intro i hi
    rw [read "zero" i (by decide)]
    have h := bound 3 (⟨i,hi⟩,()) (by trivial)
    change base.mem "zero" i=Cell.mk (α := SourceGraph.Node α) .int (v 3 (⟨i,hi⟩,())) at h
    rw [h]
    simpa only [mapCell,mapValue,Cell.read,operands,dif_pos hi,dite_true] using (domain.2.1 ⟨i,hi⟩).symm

/-- The reset is checked as an actual typed input cell, not an unbound seed. -/
theorem initial_zero (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (domain : IO.domain x.shape o v) (base : RegisterState (SourceGraph.Node α))
    (bound : (IO.programIO x t o rest lanes).InputCells v base)
    (initial : (IO.programIO x t o rest lanes).RawOutputs)
    (output : (IO.programIO x t o rest lanes).OutputCells v initial base)
    (j : TileIndex [x.shape.m,x.shape.n]) : initial ⟨0,j⟩=.value o.zero := by
  have cap := rowMajor_bound x.shape.m x.shape.n j.1.val j.2.1.val j.1.isLt j.2.1.isLt
  have h := bound 4 (⟨j.1.val*x.shape.n+j.2.1.val,cap⟩,()) (by trivial)
  have out := output ⟨0,j⟩ (by trivial)
  change base.mem "out" (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real
    (.value (v 4 (⟨j.1.val*x.shape.n+j.2.1.val,cap⟩,()))) at h
  change base.mem "out" (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real (initial ⟨0,j⟩) at out
  rw [domain.2.2] at h
  have same := out.symm.trans h
  exact congrArg (fun c=>c.read .real) same

#axiomsClean operand_binding
#axiomsClean initial_zero
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
