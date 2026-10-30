import bench.tritonbench_g.cache_transform_triton.Proofs.IO

namespace VeriTile.Bench.Repairs.CacheTransform.IOProof
open VeriTile.Triton FP.Structural
noncomputable section

/-- Whole-grid IO assembled from the existing source result and the actual
trace-safety proof. The input mode explicitly retains exact metadata/byte tags. -/
theorem implements {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit)
    (n capacity : Nat) :
    (cacheIO M p n capacity).ImplementsOnCells (CacheIO.Domain M p n capacity)
      (CacheIO.formula M p n capacity) := by
  apply ProgramIO.ImplementsOnCells.intro
  · intro xs domain bounds bounded initial cells
    exact safe_grid M p start block grid initial n capacity
      (fun q => (CacheIO.row M p n capacity xs q).toNat) (CacheIO.payload M p n capacity xs)
      (inputs M p n capacity xs domain initial cells) bounds
      (capacities M p n capacity xs bounds bounded) _ (Nat.le_refl _)
  · intro xs domain bounds bounded initial cells
    obtain ⟨t,ran,values,frame⟩ := KernelInterface.complete M p start block grid initial n capacity
      (fun q => (CacheIO.row M p n capacity xs q).toNat) (CacheIO.payload M p n capacity xs)
      (inputs M p n capacity xs domain initial cells)
    refine ⟨t,?_,?_,?_⟩
    · exact (jobs_launch M p initial _).trans ran
    · intro o
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      intro j _
      change mapCell id (t.mem p.output (j.1.val*1+0)) = _
      simp only [Nat.mul_one,Nat.add_zero]
      rw [values _ j.1.isLt]
      rfl
    · intro r address outside
      apply frame r address
      by_cases hr : r=p.output
      · right
        by_contra hn
        have lt : address<p.count := by omega
        have impossible := outside.1 (0 : Fin 1) (⟨address,lt⟩,PUnit.unit) trivial
        change r≠p.output ∨ address≠address*1+0 at impossible
        simpa [hr] using impossible
      · exact Or.inl hr

end
end VeriTile.Bench.Repairs.CacheTransform.IOProof
