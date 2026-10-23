import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.RopeTransform
open VeriTile.Triton FP.Structural KernelContract
noncomputable section

def rawRealValue (r : Request) (s : State ℝ) (key : Bool)
    (row head : Nat) (second : Bool) (lane : Nat) : ℝ :=
  let pair := realPair
    ((s.mem (region key) (r.offset key row head lane)).read .fp32)
    ((s.mem (region key) (r.offset key row head (lane+r.half))).read .fp32)
    ((s.mem "C" (row%r.sequence*r.half+lane)).read .fp32)
    ((s.mem "S" (row%r.sequence*r.half+lane)).read .fp32)
  if second then pair.2 else pair.1

/-- Complete forward source grid over raw memory and geometry, with the independent
rotation polynomial and every unwritten memory cell. -/
theorem rope_forward_raw_correct (r : Request) (s : State ℝ) :
    ∃ final,Grid.sourceRun realAlgebra r s r.rows=some final ∧
      (∀ key row head second lane,row<r.rows → head<r.heads key → lane<r.half →
        final.mem (region key) (r.offset key row head (r.column second lane))=
          Cell.mk .fp32 (rawRealValue r s key row head second lane)) ∧
      Grid.Frame r r.rows s final := by
  obtain ⟨ran,outputs,frame⟩ := Grid.complete realAlgebra r s
  refine ⟨Grid.run realAlgebra r s r.rows,ran,?_,frame⟩
  simpa only [value,scalar_real,rawRealValue] using outputs

/-- Empty raw geometry has an empty source grid. -/
theorem rope_forward_raw_no_launch {α : Type} [Inhabited α] (M : Algebra α)
    (r : Request) (s : State α) (empty : r.batches=0 ∨ r.sequence=0) :
    r.rows=0 ∧ Grid.sourceRun M r s r.rows=some s := by
  have zero : r.rows=0 := by rcases empty with h|h <;> simp [Request.rows,h]
  exact ⟨zero,by rw [zero]; rfl⟩
/-- The selected native graph depends only on input words; register tags are
cleared by the original casts. -/
theorem scalar_native_raw (cfg : FP.Config) (a b c s : Tagged) (second : Bool) :
    (scalar (nativeAlgebra cfg) a b c s second).word=
      nativePair cfg a.word b.word c.word s.word second := by
  cases second <;> rfl

/-- Complete FP32 source grid, independent native rotation words and full frame,
over raw state and launch geometry. -/
theorem rope_forward_raw_fp32 (cfg : FP.Config) (r : Request) (s : State Tagged) :
    ∃ final,Grid.sourceRun (nativeAlgebra cfg) r s r.rows=some final ∧
      (∀ key row head second lane,row<r.rows → head<r.heads key → lane<r.half →
        ((final.mem (region key) (r.offset key row head (r.column second lane))).read .fp32).word=
          nativePair cfg
            (((s.mem (region key) (r.offset key row head lane)).read .fp32).word)
            (((s.mem (region key) (r.offset key row head (lane+r.half))).read .fp32).word)
            (((s.mem "C" (row%r.sequence*r.half+lane)).read .fp32).word)
            (((s.mem "S" (row%r.sequence*r.half+lane)).read .fp32).word) second) ∧
      Grid.Frame r r.rows s final := by
  obtain ⟨ran,outputs,frame⟩ := Grid.complete (nativeAlgebra cfg) r s
  refine ⟨Grid.run (nativeAlgebra cfg) r s r.rows,ran,?_,frame⟩
  intro key row head second lane hr hh hl
  rw [outputs key row head second lane hr hh hl,Cell.read_same]
  exact scalar_native_raw _ _ _ _ _ _

#axiomsClean rope_forward_raw_fp32
#axiomsClean rope_forward_raw_correct
#axiomsClean rope_forward_raw_no_launch
end
end VeriTile.Bench.Repairs.RopeTransform
