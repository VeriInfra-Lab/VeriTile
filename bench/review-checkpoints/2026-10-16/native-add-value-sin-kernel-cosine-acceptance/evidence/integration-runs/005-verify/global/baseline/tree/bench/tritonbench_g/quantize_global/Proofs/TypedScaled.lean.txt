import bench.tritonbench_g.quantize_global.QuantizeGlobal
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal.TypedScaled
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- The already accepted pre-llrint scaled-store stage, with actual fp32 storage
and compute precision. This does not implement the final integer quantization. -/
def source (X AbsmaxInv Out : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, AbsmaxInv = tl.float32, Out = tl.float32
  pid = tl.program_id(0)
  block_start = pid * $(B)
  offsets = block_start + tl.arange(0, $(B))
  mask = offsets < $(N)
  x = tl.load(X + offsets, mask=mask).to(tl.float32)
  absmax_inv = tl.load(AbsmaxInv).to(tl.float32)
  output = 127.0 * (x * absmax_inv)
  tl.store(Out + offsets, output, mask=mask)
}

abbrev Word := FP.Value .fp32
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def scale127 : Word := FP.Value.ofBits .fp32 0x42fe0000

noncomputable def backend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest cfg
  { base with literal := fun p d x =>
      if FP.Software.fp32Context p d && decide (x = 127) then scale127
      else base.literal p d x }

def offset (s : State Word) (B : Nat) (i : TileIndex [B]) := s.pids 0 * B + i.1.val

def value (cfg : FP.Config) (s : State Word) (X AbsmaxInv : RegionName)
    (B : Nat) (i : TileIndex [B]) : Word :=
  FP.mul cfg scale127 (FP.mul cfg ((s.mem X (offset s B i)).read .real)
    ((s.mem AbsmaxInv 0).read .real))

def written (cfg : FP.Config) (s : State Word) (X AbsmaxInv Out : RegionName)
    (N B : Nat) : State Word :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if offset s B i < N then t.write Out (offset s B i) (.mk .real (value cfg s X AbsmaxInv B i))
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
    (X AbsmaxInv Out : RegionName) (N B : Nat) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source X AbsmaxInv Out N B) s = some t ∧
      t.mem = (written cfg s X AbsmaxInv Out N B).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [source, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, backend, FP.Software.scalarFP32,
    FP.Software.fp32Context, State.setReg, TileShape.axisDim,
    written, value, offset, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
    numericLt, natLt, Option.bind]
  simp only [fold_memory]
  simp +contextual [show (127.0 : ℝ) = 127 by norm_num]
  all_goals rfl

theorem offset_injective (s : State Word) (B : Nat) : Function.Injective (offset s B) := by
  intro i j h
  have hf : i.1 = j.1 := Fin.ext (Nat.add_left_cancel h)
  exact Prod.ext hf (Subsingleton.elim _ _)

theorem source_correct (rest : Algebra Word) (cfg : FP.Config) (s : State Word)
    (X AbsmaxInv Out : RegionName) (N B : Nat) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source X AbsmaxInv Out N B) s = some t ∧
      (∀ i, offset s B i < N → t.mem Out (offset s B i) = .mk .real (value cfg s X AbsmaxInv B i)) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i, offset s B i < N → o ≠ offset s B i) → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run rest cfg s X AbsmaxInv Out N B
  refine ⟨t, he, ?_, ?_⟩
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback Out (offset s B) _ (fun i => offset s B i < N)
      s i hi (offset_injective s B)
  · intro r o hmiss
    rw [hm]
    unfold written
    apply State.masked_scatter_frame
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)

#axiomsClean source_run
#axiomsClean offset_injective
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.QuantizeGlobal.TypedScaled
