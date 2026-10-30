import bench.tritonbench_g.quantize_kv_copy.QuantizeKvCopy
import bench.tritonbench_g.quantize_kv_copy.Proofs.FP32Profile
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.IntegerConversionTotal

/-! The accepted grouped quantizer, using the supplied fp32 input/scale and
signed-int8 output profile. Arithmetic uses actual binary32 encodings, each
quotient is rounded separately, and the integer conversion checks its exact
finite toward-zero domain. No GPU reduction-tree or lowering equivalence is
inferred merely from this executable bit interpretation. -/
namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Parameters where
  (input output scales : RegionName)
  destination : Region .nat
  (strideKbs strideKh strideKg strideObs strideOh strideOg strideSbs strideSh
    groups height width : Nat)

def source (p : Parameters) : ComputeKernel :=
  let K := p.input
  let DestLoc := p.destination
  let Out := p.output
  let OutScale := p.scales
  triton {
  tl.region K = tl.float32, Out = tl.int8, OutScale = tl.float32
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(p.height))
  offs_d = tl.arange(0, $(p.width))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(p.strideKbs) + cur_head * $(p.strideKh) +
      offs_g[:, None] * $(p.strideKg) + offs_d[None, :],
    mask=offs_g[:, None] < $(p.groups), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = (tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty)
  q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
  o_ptrs = Out + dest_index * $(p.strideObs) + cur_head * $(p.strideOh) +
    offs_g[:, None] * $(p.strideOg) + offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(p.strideSbs) + cur_head * $(p.strideSh) + offs_g
  tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < $(p.groups))
  tl.store(os_ptrs, data_scale, mask=offs_g < $(p.groups))
}


def active (p : Parameters) (i : TileIndex [p.height, p.width]) : Prop := i.1.val < p.groups
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ < _))
def inputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  s.pids 0 * p.strideKbs + s.pids 1 * p.strideKh + i.1.val * p.strideKg + i.2.1.val

def loaded (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Word :=
  if active p i then (s.mem p.input (inputOffset p s i)).read .real else FP.Value.zero .fp32

def rowMaximum (p : Parameters) (s : State Word) (g : Fin p.height) : Word :=
  maximum (fun d : Fin p.width => (loaded p s (g, d, PUnit.unit)).abs)

def scale (cfg : FP.Config) (p : Parameters) (s : State Word) (g : Fin p.height) : Word :=
  FP.div cfg (rowMaximum p s g) (FP.Value.ofBits .fp32 0x42fe0000)

def quotient (cfg : FP.Config) (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Word :=
  FP.div cfg (loaded p s i) (scale cfg p s i.1)

def InputDefined (cfg : FP.Config) (p : Parameters) (s : State Word) : Prop :=
  ∀ i, active p i → (toSigned 8 (quotient cfg p s i)).isSome

def expected (cfg : FP.Config) (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Int :=
  (toSigned 8 (quotient cfg p s i)).getD 0

def destination (p : Parameters) (s : State Word) : Nat :=
  (s.mem p.destination.cast (s.pids 0)).read .nat

def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  destination p s * p.strideObs + s.pids 1 * p.strideOh + i.1.val * p.strideOg + i.2.1.val

def scaleOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height]) : Nat :=
  destination p s * p.strideSbs + s.pids 1 * p.strideSh + i.1.val

def valuesWritten (cfg : FP.Config) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height, p.width]).foldl (fun t i => if active p i then
    t.write p.output (outputOffset p s i) (.mk .int (expected cfg p s i)) else t) s

def written (cfg : FP.Config) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height]).foldl (fun t i => if i.1.val < p.groups then
    t.write p.scales (scaleOffset p s i) (.mk .real (scale cfg p s i.1)) else t)
    (valuesWritten cfg p s)

/-- Every statement of the original grouped stage executes. A padded row may
compute 0/0 before its store is masked; its integer result stays arbitrary. -/
def execution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) (hw : 0 < p.width) :
    {t : State Word // FP.Structural.exec (backend rest cfg u) (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run,
    FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.axisDim, hw]
  try rfl

theorem conversion_defined (u : Word → Int) (x : Word) (h : (toSigned 8 x).isSome) :
    totalSigned u 8 x = (toSigned 8 x).getD 0 := by
  cases hx : toSigned 8 x with
  | none => simp [hx] at h
  | some z => simp [totalSigned, hx]

theorem execution_memory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) (hw : 0 < p.width) (h : InputDefined cfg p s) :
    (execution rest cfg u p s hw).val.mem = (written cfg p s).mem := by
  simp +contextual [execution, written, valuesWritten, masked_scatter_memory,
    active, inputOffset, outputOffset, scaleOffset, destination, expected,
    loaded, scale, rowMaximum, quotient, reduction,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    TileShape.insertAxisIndex, TileShape.replaceAxisIndex,
    show (0.0 : ℝ) = 0 by norm_num, show (127.0 : ℝ) = 127 by norm_num]
  congr 1
  congr 1
  funext mem i
  by_cases hi : active p i
  · have hc := conversion_defined (u 8) (quotient cfg p s i) (h i hi)
    simp only [active] at hi
    simp [quotient, scale, rowMaximum, loaded, inputOffset, active, hi] at hc
    simp only [hi, ↓reduceIte]
    funext r o
    split_ifs <;> try rfl
    congr 1
  · simp only [active] at hi
    simp only [hi, ↓reduceIte]


/-- Independent maximum specification over the abs-transformed input bits. -/
theorem maximum_spec {n : Nat} (x : Fin n → Word) (hn : 0 < n) :
    MatrixReduction.MachineMax.IsMaximum .fp32 (fun i => (x i).bits) (maximum x).bits := by
  simpa [maximum, MatrixReduction.MachineMax.rowMaximum, hn] using
    MatrixReduction.MachineMax.best_spec .fp32 (fun i => (x i).bits) hn

theorem expected_spec (cfg : FP.Config) (p : Parameters) (s : State Word)
    (h : InputDefined cfg p s) (i : TileIndex [p.height, p.width]) (hi : active p i) :
    ∃ q, (quotient cfg p s i).toRat? = some q ∧
      TowardZero q (expected cfg p s i) ∧ SignedRange 8 (expected cfg p s i) := by
  have hd := h i hi
  cases hx : toSigned 8 (quotient cfg p s i) with
  | none => simp [hx] at hd
  | some z => simpa [expected, hx] using (toSigned_spec 8 (quotient cfg p s i) z).mp hx

theorem value_readback (cfg : FP.Config) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) (hi : active p i)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    (written cfg p s).mem p.output (outputOffset p s i) = .mk .int (expected cfg p s i) := by
  unfold written
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hregions)]
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem scale_readback (cfg : FP.Config) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height]) (hi : i.1.val < p.groups) :
    (written cfg p s).mem p.scales (scaleOffset p s i) = .mk .real (scale cfg p s i.1) := by
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b _ _ hab
  have hv : a.1.val = b.1.val := by simpa [scaleOffset] using hab
  exact Prod.ext (Fin.ext hv) (Subsingleton.elim _ _)

theorem frame (cfg : FP.Config) (p : Parameters) (s : State Word) (r : RegionName) (o : Nat)
    (hv : r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i → o ≠ outputOffset p s i)
    (hs : r ≠ p.scales ∨ ∀ i : TileIndex [p.height], i.1.val < p.groups → o ≠ scaleOffset p s i) :
    (written cfg p s).mem r o = s.mem r o := by
  unfold written valuesWritten
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (hs.imp id (fun h i _ => h i))]
  exact State.masked_scatter_frame _ _ _ _ _ _ _ (hv.imp id (fun h i _ => h i)) _

theorem undefined_policy_independent (rest : Algebra Word) (cfg : FP.Config)
    (u v : Nat → Word → Int) (p : Parameters) (s : State Word) (hw : 0 < p.width)
    (h : InputDefined cfg p s) :
    (execution rest cfg u p s hw).val.mem = (execution rest cfg v p s hw).val.mem := by
  rw [execution_memory rest cfg u p s hw h, execution_memory rest cfg v p s hw h]

/-- Successful original grouped execution, exact separately rounded scale and
quotient, a source-bit maximum certificate, signed-int8 rational truncation,
and preservation outside both active write sets. -/
theorem correctness (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) (hw : 0 < p.width) (h : InputDefined cfg p s)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (backend rest cfg u) (source p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], active p i →
        t.mem p.output (outputOffset p s i) = .mk .int (expected cfg p s i) ∧
        ∃ q, (quotient cfg p s i).toRat? = some q ∧
          TowardZero q (expected cfg p s i) ∧ SignedRange 8 (expected cfg p s i)) ∧
      (∀ i : TileIndex [p.height], i.1.val < p.groups →
        t.mem p.scales (scaleOffset p s i) = .mk .real (scale cfg p s i.1) ∧
        MatrixReduction.MachineMax.IsMaximum .fp32
          (fun d : Fin p.width => (loaded p s (i.1, d, PUnit.unit)).abs.bits)
          (rowMaximum p s i.1).bits) ∧
      (∀ r o,
        (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i → o ≠ outputOffset p s i) →
        (r ≠ p.scales ∨ ∀ i : TileIndex [p.height], i.1.val < p.groups → o ≠ scaleOffset p s i) →
        t.mem r o = s.mem r o) := by
  refine ⟨(execution rest cfg u p s hw).val, (execution rest cfg u p s hw).property, ?_, ?_, ?_⟩
  · intro i hi
    rw [execution_memory rest cfg u p s hw h]
    exact ⟨value_readback cfg p s i hi hregions hinj, expected_spec cfg p s h i hi⟩
  · intro i hi
    rw [execution_memory rest cfg u p s hw h]
    exact ⟨scale_readback cfg p s i hi, maximum_spec _ hw⟩
  · intro r o hv hs
    rw [execution_memory rest cfg u p s hw h]
    exact frame cfg p s r o hv hs


#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean maximum_spec
#axiomsClean expected_spec
#axiomsClean value_readback
#axiomsClean scale_readback
#axiomsClean frame
#axiomsClean undefined_policy_independent
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize
