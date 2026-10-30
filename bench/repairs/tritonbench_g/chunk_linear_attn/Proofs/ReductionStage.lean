import bench.repairs.tritonbench_g.chunk_linear_attn.ReductionIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.LibraryReduction
open VeriTile Triton FP.Structural
noncomputable section

theorem leaves {n : Nat} (t : LibrarySum.Tree n) : (tree t).leaves = t.leaves := by
  induction t with
  | zero => rfl
  | input i => rfl
  | add a b ha hb => simp only [tree, FP32AddTree.leaves, LibrarySum.Tree.leaves, ha, hb]

theorem evaluate {n : Nat} {α : Type} (M : Algebra α) (t : LibrarySum.Tree n)
    (x : Fin n → α) :
    (tree t).eval M x = t.eval (M.literal (some .fp32) .real 0)
      (M.binary (some .fp32) .real .add) (fun i => M.fp32Load (x i)) := by
  induction t with
  | zero => rfl
  | input i => rfl
  | add a b ha hb => simp only [tree, FP32AddTree.eval, LibrarySum.Tree.eval, ha, hb]

variable {α : Type} [Inhabited α]

theorem value_eq (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (s : State α) (o : Fin (valueCount p)) :
    (sumStage M p selection).value s o = cast M (sumValue M p selection s.mem o) := by
  unfold FP32SumStage.value sumStage plan sumValue LibrarySum.evaluate cast
  rw [evaluate]

theorem run_mem (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (s : State α) :
    ((sumStage M p selection).run s).mem = hostSumMemory M p selection s.mem s.mem := by
  have indices : TileShape.allIndices [valueCount p] =
      (List.ofFn (fun o : Fin (valueCount p) => o)).map (fun o => (o, PUnit.unit)) := by
    simp only [TileShape.allIndices, List.map_cons, List.map_nil, List.ofFn_id]
    rw [← List.map_eq_flatMap]
    congr 1
  have fold : ∀ (xs : List (Fin (valueCount p))) (current : State α),
      (xs.foldl (fun t o => t.write "valueGradient" o.val
        (.mk .real (cast M (sumValue M p selection s.mem o)))) current).mem =
      scatter (xs.map (fun o => (o, PUnit.unit))) "valueGradient" (fun _ => True)
        (fun i => i.1.val) (fun i => .mk .real (cast M (sumValue M p selection s.mem i.1)))
        current.mem := by
    intro xs
    induction xs with
    | nil => intro current; rfl
    | cons o rest ih =>
        intro current
        simp only [List.foldl_cons, List.map_cons]
        rw [ih]
        rfl
  unfold FP32SumStage.run hostSumMemory
  simp only [value_eq]
  rw [indices]
  exact fold _ _

/-- Safety uses only the exact memory extents of the finite selected sum. -/
theorem safe (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (bounds : RegionName → Nat)
    (inputRoom : keyBlocks p * valueCount p ≤ bounds "valuePartials")
    (outputRoom : valueCount p ≤ bounds "valueGradient") :
    (sumStage M p selection).Safe bounds := by
  constructor
  · intro o k
    exact lt_of_lt_of_le
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt k.isLt o.isLt) inputRoom
  · intro o
    exact lt_of_lt_of_le o.isLt outputRoom

#axiomsClean leaves
#axiomsClean evaluate
#axiomsClean value_eq
#axiomsClean run_mem
#axiomsClean safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.LibraryReduction
