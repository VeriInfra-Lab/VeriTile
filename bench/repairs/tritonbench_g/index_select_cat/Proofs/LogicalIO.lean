import bench.repairs.tritonbench_g.index_select_cat.IO
import bench.repairs.tritonbench_g.index_select_cat.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.IndexSelectCat.LogicalIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section

theorem input_offset [Inhabited α] (p : Layout) (rowPid colPid : Nat)
    (indices : Fin p.BI → Int) (s : State α)
    (bound : ∀ i, index p rowPid i < p.rows → (s.mem p.index (index p rowPid i)).read .int = indices i)
    (i : TileIndex [p.BI, p.BC]) (active : Active p rowPid colPid i) :
    TypedGather.inputOffset p (initial rowPid colPid s) i = read p colPid indices i := by
  have binding : (s.mem p.index (rowPid * p.BI + i.1.val)).read .int = indices i.1 :=
    bound i.1 active.1
  simp [TypedGather.inputOffset, TypedGather.row, TypedGather.indexOffset,
    TypedGather.column, initial, column, read, binding]

theorem complete [Inhabited α] (rest : Algebra α) (p : Layout) (rowPid colPid : Nat)
    (indices : Fin p.BI → Int) (xs : TileIndex [p.BI, p.BC] → α) (s : State α)
    (metadata : ∀ i, index p rowPid i < p.rows → (s.mem p.index (index p rowPid i)).read .int = indices i)
    (payload : ∀ i, Active p rowPid colPid i → (s.mem p.source (read p colPid indices i)).read .real = xs i)
    (unique : Unique p rowPid colPid) :
    ∃ t, FP.Structural.exec (copyBackend rest) (kernel p) (initial rowPid colPid s) = some t ∧
      (∀ i, Active p rowPid colPid i → t.mem p.output (write p rowPid colPid i) = .mk .real (xs i)) ∧
      (∀ r o, (∀ i, Active p rowPid colPid i → r ≠ p.output ∨ o ≠ write p rowPid colPid i) →
        t.mem r o = s.mem r o) := by
  have active (i) : TypedGather.active p (initial rowPid colPid s) i ↔ Active p rowPid colPid i := by
    simp [TypedGather.active, TypedGather.indexOffset, TypedGather.column, initial, Active, index, column]
  have output (i) : TypedGather.outputOffset p (initial rowPid colPid s) i = write p rowPid colPid i := by
    simp [TypedGather.outputOffset, TypedGather.indexOffset, TypedGather.column, initial, write, index, column]
  obtain ⟨t, run, values, frame⟩ := RawKernel.complete rest p (initial rowPid colPid s)
    (fun a b ha hb eq => unique a b ((active a).mp ha) ((active b).mp hb)
      ((output a).symm.trans (eq.trans (output b))))
  refine ⟨t, run, ?_, ?_⟩
  · intro i hi
    rw [← output, values i ((active i).mpr hi)]
    unfold TypedGather.loaded
    rw [input_offset p rowPid colPid indices s metadata i hi]
    exact congrArg (Cell.mk .real) (payload i hi)
  · intro r o outside
    apply frame
    by_cases hr : r ≠ p.output
    · exact Or.inl hr
    · right
      intro i hi
      rw [output i]
      exact (outside i ((active i).mp hi)).resolve_left hr

theorem safety [Inhabited α] (rest : Algebra α) (p : Layout) (rowPid colPid : Nat)
    (indices : Fin p.BI → Int) (s : State α) (bounds : RegionName → Nat)
    (metadata : ∀ i, index p rowPid i < p.rows → (s.mem p.index (index p rowPid i)).read .int = indices i)
    (indexBound : ∀ i, index p rowPid i < p.rows → index p rowPid i < bounds p.index)
    (readBound : ∀ i, Active p rowPid colPid i → read p colPid indices i < bounds p.source)
    (writeBound : ∀ i, Active p rowPid colPid i → write p rowPid colPid i < bounds p.output) :
    FP.Structural.TraceSafe (copyBackend rest) bounds (kernel p) (initial rowPid colPid s) := by
  apply KernelSafety.trace rest p bounds
  · simpa [initial, index] using indexBound
  · intro i active
    have hi : Active p rowPid colPid i := by
      simpa [TypedGather.active, TypedGather.indexOffset, TypedGather.column,
        initial, Active, index, column] using active
    rw [input_offset p rowPid colPid indices s metadata i hi]
    exact readBound i hi
  · simpa [TypedGather.active, TypedGather.indexOffset, TypedGather.column,
      TypedGather.outputOffset, initial, Active, write, index, column] using writeBound

end
end VeriTile.Bench.Repairs.IndexSelectCat.LogicalIO
