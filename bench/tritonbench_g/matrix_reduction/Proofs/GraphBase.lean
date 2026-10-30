import bench.tritonbench_g.matrix_reduction.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Task-local typed channels and scatter lemmas needed by both actual programs.
The namespace preserves compatibility with earlier concrete-model consumers. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.ConcreteCorrect
open VeriTile.Triton FP.Structural TypedSource
noncomputable section
def channel : Storage → FloatDType | .fp16 => .fp16 | .fp32 => .real

theorem fold_memory {α ι : Type} (l : List ι) (Y : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (s : State α) :
    (l.foldl (fun t i => t.write Y (off i) (value i)) s).mem =
      l.foldl (fun mem i => fun r o =>
        if r = Y ∧ o = off i then value i else mem r o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => simp only [List.foldl_cons, ih]; rfl

theorem lane_injective (M : Nat) :
    Function.Injective (fun i : TileIndex [M] => i.1.val) := by
  rintro ⟨i, u⟩ ⟨j, v⟩ h
  have : i = j := Fin.ext h
  subst j
  cases u; cases v
  rfl


#axiomsClean fold_memory
#axiomsClean lane_injective
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.ConcreteCorrect
