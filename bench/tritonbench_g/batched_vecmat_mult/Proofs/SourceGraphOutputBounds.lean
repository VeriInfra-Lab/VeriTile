import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphAccess

/-! Output bounds and uniqueness do not depend on a K-loop index. In
particular K=0 still executes the final store when output tiles are nonempty.
These are natural element-address properties of the actual logical grid;
native byte arithmetic, allocation and instruction interleavings are separate. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨Expr.zero⟩

theorem output_address_in_declared_extent (p : Parameters)
    (m : Fin (gridM p)) (n : Fin (gridN p)) (i : Fin p.BM) (j : Fin p.BN) :
    (m.val*p.BM+i.val)*p.N + (n.val*p.BN+j.val) < p.M*p.N := by
  exact row_major_bound (tiled_coordinate_lt p.M p.BM m i)
    (tiled_coordinate_lt p.N p.BN n j)

private theorem equal_flat_parts {r₁ r₂ c₁ c₂ width : Nat}
    (hc₁ : c₁ < width) (hc₂ : c₂ < width)
    (same : r₁*width+c₁ = r₂*width+c₂) : r₁ = r₂ ∧ c₁ = c₂ := by
  have cols := congrArg (fun value => value % width) same
  dsimp only at cols
  rw [Nat.mul_add_mod_of_lt hc₁, Nat.mul_add_mod_of_lt hc₂] at cols
  have products : r₁*width = r₂*width := by omega
  exact ⟨Nat.mul_right_cancel (by omega : 0 < width) products, cols⟩

/-- Across all launched programs, each logical output element has at most
one writing lane. This holds even when the K-loop is empty. -/
theorem output_address_unique (p : Parameters)
    (m₁ m₂ : Fin (gridM p)) (n₁ n₂ : Fin (gridN p))
    (i₁ i₂ : Fin p.BM) (j₁ j₂ : Fin p.BN)
    (same : (m₁.val*p.BM+i₁.val)*p.N + (n₁.val*p.BN+j₁.val) =
      (m₂.val*p.BM+i₂.val)*p.N + (n₂.val*p.BN+j₂.val)) :
    m₁ = m₂ ∧ n₁ = n₂ ∧ i₁ = i₂ ∧ j₁ = j₂ := by
  obtain ⟨rows,cols⟩ := equal_flat_parts
    (tiled_coordinate_lt p.N p.BN n₁ j₁)
    (tiled_coordinate_lt p.N p.BN n₂ j₂) same
  obtain ⟨hm,hi⟩ := equal_flat_parts i₁.isLt i₂.isLt rows
  obtain ⟨hn,hj⟩ := equal_flat_parts j₁.isLt j₂.isLt cols
  exact ⟨Fin.ext hm,Fin.ext hn,Fin.ext hi,Fin.ext hj⟩

/-- The bound applies directly to the address used by the actual source
in each state created by `programState`; there is no c : Fin (K/BK) premise. -/
theorem actual_program_output_in_bounds (p : Parameters) (initial : State Word)
    (mem : Memory) (k : Nat) (hk : k < gridCount p) (i : TileIndex [p.BM,p.BN]) :
    outputOffset p (programState p initial mem k) i < p.M*p.N := by
  obtain ⟨hm,hn⟩ := coordinate_bounds p k hk
  have bound := output_address_in_declared_extent p
    ⟨k % gridM p,hm⟩ ⟨k / gridM p,hn⟩ i.1 i.2.1
  simpa only [outputOffset,row,col,programState,↓reduceIte] using bound

/-- Equality of two actual output addresses determines both the program ID
and lane. Thus all actual logical output writes are pairwise distinct. -/
theorem actual_program_output_unique (p : Parameters) (initial : State Word)
    (leftMemory rightMemory : Memory) (k l : Nat)
    (hk : k < gridCount p) (hl : l < gridCount p)
    (i j : TileIndex [p.BM,p.BN])
    (same : outputOffset p (programState p initial leftMemory k) i =
      outputOffset p (programState p initial rightMemory l) j) : k = l ∧ i = j := by
  obtain ⟨hkm,hkn⟩ := coordinate_bounds p k hk
  obtain ⟨hlm,hln⟩ := coordinate_bounds p l hl
  have normalized :
      ((k % gridM p)*p.BM+i.1.val)*p.N + ((k / gridM p)*p.BN+i.2.1.val) =
      ((l % gridM p)*p.BM+j.1.val)*p.N + ((l / gridM p)*p.BN+j.2.1.val) := by
    simpa only [outputOffset,row,col,programState,↓reduceIte] using same
  obtain ⟨hm,hn,hi,hj⟩ := output_address_unique p
    ⟨k % gridM p,hkm⟩ ⟨l % gridM p,hlm⟩
    ⟨k / gridM p,hkn⟩ ⟨l / gridM p,hln⟩ i.1 j.1 i.2.1 j.2.1 normalized
  have eqmod : k % gridM p = l % gridM p := congrArg Fin.val hm
  have eqdiv : k / gridM p = l / gridM p := congrArg Fin.val hn
  have reconstruct := Nat.mod_add_div k (gridM p)
  rw [eqmod,eqdiv] at reconstruct
  exact ⟨reconstruct.symm.trans (Nat.mod_add_div l (gridM p)),
    Prod.ext hi (Prod.ext hj (Subsingleton.elim _ _))⟩

#axiomsClean output_address_in_declared_extent
#axiomsClean output_address_unique
#axiomsClean actual_program_output_in_bounds
#axiomsClean actual_program_output_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
