import bench.tritonbench_g.quant_transpose_kernel.Kernels
import VeriTile.Triton.Float.ScalarBackend
local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel.TypedScaled
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

abbrev Word := FP.Value .fp32
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def scale127 : Word := FP.Value.ofBits .fp32 0x42fe0000

noncomputable def backend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest cfg
  { base with literal := fun p d x =>
      if FP.Software.fp32Context p d && decide (x = 127) then scale127
      else base.literal p d x }

def row (s : State Word) (BM : Nat) (i : Fin BM) := s.pids 0 * BM + i.val
def col (s : State Word) (BN : Nat) (j : Fin BN) := s.pids 1 * BN + j.val

def active (s : State Word) (M N BM BN : Nat) (i : TileIndex [BM, BN]) : Prop :=
  row s BM i.1 < M ∧ col s BN i.2.1 < N
instance (s : State Word) (M N BM BN : Nat) (i : TileIndex [BM, BN]) :
    Decidable (active s M N BM BN i) := inferInstanceAs (Decidable (_ ∧ _))

def inputOffset (s : State Word) (sam san BM BN : Nat) (i : TileIndex [BM, BN]) :=
  row s BM i.1 * sam + col s BN i.2.1 * san

def outputOffset (s : State Word) (sbn sbm BM BN : Nat) (i : TileIndex [BM, BN]) :=
  row s BM i.1 * sbm + col s BN i.2.1 * sbn

/-- Two rounded multiplies in the source order. The target expression does not
refer to the AST, its evaluator, or a reduction/execution result. -/
def value (cfg : FP.Config) (s : State Word) (A AbsmaxInv : RegionName)
    (sam san BM BN : Nat) (i : TileIndex [BM, BN]) : Word :=
  FP.mul cfg scale127 (FP.mul cfg
    ((s.mem A (inputOffset s sam san BM BN i)).read .real)
    ((s.mem AbsmaxInv 0).read .real))

def written (cfg : FP.Config) (s : State Word) (A AbsmaxInv B : RegionName)
    (sam san sbn sbm M N BM BN : Nat) : State Word :=
  (TileShape.allIndices [BM, BN]).foldl (fun t i =>
    if active s M N BM BN i then
      t.write B (outputOffset s sbn sbm BM BN i) (.mk .real (value cfg s A AbsmaxInv sam san BM BN i))
    else t) s

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

theorem source_run (rest : Algebra Word) (cfg : FP.Config) (s : State Word)
    (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source A AbsmaxInv B sam san sbn sbm M N BM BN) s = some t ∧
      t.mem = (written cfg s A AbsmaxInv B sam san sbn sbm M N BM BN).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [source, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, backend, FP.Software.scalarFP32,
    FP.Software.fp32Context, State.setReg, TileShape.axisDim,
    written, value, active, inputOffset, outputOffset, row, col,
    ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
    numericLt, natLt, Option.bind]
  simp only [fold_memory]
  simp +contextual [show (127.0 : ℝ) = 127 by norm_num]
  all_goals rfl

theorem source_correct (rest : Algebra Word) (cfg : FP.Config) (s : State Word)
    (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat)
    (hOutInj : Function.Injective (outputOffset s sbn sbm BM BN)) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source A AbsmaxInv B sam san sbn sbm M N BM BN) s = some t ∧
      (∀ i, active s M N BM BN i →
        t.mem B (outputOffset s sbn sbm BM BN i) = .mk .real (value cfg s A AbsmaxInv sam san BM BN i)) ∧
      (∀ r o, (r ≠ B ∨ ∀ i, active s M N BM BN i → o ≠ outputOffset s sbn sbm BM BN i) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run rest cfg s A AbsmaxInv B sam san sbn sbm M N BM BN
  refine ⟨t, he, ?_, ?_⟩
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback B (outputOffset s sbn sbm BM BN) _
      (active s M N BM BN) s i hi hOutInj
  · intro r o hmiss
    rw [hm]
    unfold written
    apply State.masked_scatter_frame
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)

#axiomsClean source_run
#axiomsClean source_correct

end
end VeriTile.Bench.TritonBenchG.QuantTransposeKernel.TypedScaled
