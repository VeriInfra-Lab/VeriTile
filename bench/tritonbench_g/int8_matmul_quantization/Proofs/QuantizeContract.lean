import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeComplete
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
local instance contractWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def globalIndex (p : Parameters) (hw : 0<p.blockColumns) (i : Fin p.blockRows) (j : Nat) : Index p :=
  (i,⟨j%p.blockColumns,Nat.mod_lt _ hw⟩,PUnit.unit)

def globalQuotient (cfg : FP.Config) (p : Parameters) (s : State Word) (i : Fin p.blockRows) (j : Nat) : Word :=
  FP.div cfg ((s.mem p.input (row p s i*p.inputRowStride+j*p.inputColumnStride)).read .real) (scale cfg p s i)

def globalExpected (cfg : FP.Config) (p : Parameters) (s : State Word) (i : Fin p.blockRows) (j : Nat) : Int :=
  (toSigned 8 (globalQuotient cfg p s i j)).getD 0

theorem indexBlock (p : Parameters) (hw : 0<p.blockColumns)
    (hcover : p.columns ≤ p.steps*p.blockColumns) (j : Fin p.columns) : j.val/p.blockColumns < p.steps := by
  apply (Nat.div_lt_iff_lt_mul hw).mpr
  exact j.isLt.trans_le hcover

theorem indexActive (p : Parameters) (hw : 0<p.blockColumns)
    (i : Fin p.blockRows) (j : Fin p.columns) :
    active p (j.val/p.blockColumns) (globalIndex p hw i j.val) := by
  have h := Nat.mod_add_div j.val p.blockColumns
  have hj := j.isLt
  simp only [active, globalIndex]
  rw [Nat.mul_comm p.blockColumns] at h
  omega

theorem offsetGlobal (p : Parameters) (s : State Word) (hw : 0<p.blockColumns)
    (i : Fin p.blockRows) (j : Nat) :
    inputOffset p s (j/p.blockColumns) (globalIndex p hw i j) = row p s i*p.inputRowStride+j*p.inputColumnStride ∧
    outputOffset p s (j/p.blockColumns) (globalIndex p hw i j) = row p s i*p.outputRowStride+j*p.outputColumnStride := by
  have h : j%p.blockColumns+(j/p.blockColumns)*p.blockColumns=j := by
    simpa [Nat.mul_comm] using Nat.mod_add_div j p.blockColumns
  have haddr (rs cs : Nat) :
      row p s i*rs+(j%p.blockColumns)*cs+(j/p.blockColumns)*(p.blockColumns*cs)=row p s i*rs+j*cs := by
    calc
      _ = row p s i*rs+(j%p.blockColumns+(j/p.blockColumns)*p.blockColumns)*cs := by ring
      _ = _ := by rw [h]
  exact ⟨haddr _ _,haddr _ _⟩

theorem quotientGlobal (cfg : FP.Config) (p : Parameters) (s : State Word) (hw : 0<p.blockColumns)
    (i : Fin p.blockRows) (j : Fin p.columns) :
    quotient cfg p s (j.val/p.blockColumns) (globalIndex p hw i j.val) = globalQuotient cfg p s i j.val := by
  simp only [quotient, loaded, if_pos (indexActive p hw i j), (offsetGlobal p s hw i j.val).1]
  rfl

def Post (cfg : FP.Config) (p : Parameters) (initial final : State Word) : Prop :=
  (∀ i : Fin p.blockRows, ∀ j : Fin p.columns,
    final.mem p.output (row p initial i*p.outputRowStride+j.val*p.outputColumnStride) =
      .mk .int (globalExpected cfg p initial i j.val) ∧
      ∃ q, (globalQuotient cfg p initial i j.val).toRat?=some q ∧
        TowardZero q (globalExpected cfg p initial i j.val) ∧ SignedRange 8 (globalExpected cfg p initial i j.val)) ∧
  (∀ i : Fin p.blockRows, final.mem p.scales (scaleOffset p initial i) =
    .mk .fp16 (QuantizeKvTransform.MixedBits.narrow cfg (scale cfg p initial i))) ∧
  (∀ r o,
    (r ≠ p.output ∨ ∀ k, k<p.steps → ∀ i, active p k i → o ≠ outputOffset p initial k i) →
    (r ≠ p.scales ∨ ∀ i, o ≠ scaleOffset p initial i) → final.mem r o=initial.mem r o)

theorem correctness (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) (hw : 0<p.blockColumns)
    (hcover : p.columns ≤ p.steps*p.blockColumns) (hsep : p.input ≠ p.output)
    (hregions : p.output ≠ p.scales) (hd : InputDefined cfg p initial)
    (hinj : ∀ a b : WriteIndex p, a.1 < p.steps → b.1 < p.steps →
      active p a.1 a.2 → active p b.1 b.2 →
      outputOffset p initial a.1 a.2 = outputOffset p initial b.1 b.2 → a=b) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial = some final ∧
      Post cfg p initial final := by
  obtain ⟨final, he, hm⟩ := executionMemory rest cfg u p initial hw hsep hd
  refine ⟨final,he,?_,?_,?_⟩
  · intro i j
    rw [hm]
    have hk := indexBlock p hw hcover j
    have ha := indexActive p hw i j
    have hv := finalValues cfg p initial hregions hinj
      (j.val/p.blockColumns,globalIndex p hw i j.val) hk ha
    have hs := expectedSpec cfg p initial hd _ hk _ ha
    rw [(offsetGlobal p initial hw i j.val).2] at hv
    simpa only [expected, quotientGlobal, globalExpected] using And.intro hv hs
  · intro i
    rw [hm]
    exact finalScale cfg p initial i
  · intro r o hv hs
    rw [hm]
    exact finalFrame cfg p initial r o hv hs

theorem undefinedPolicyIndependent (rest : Algebra Word) (cfg : FP.Config)
    (u v : Nat → Word → Int) (p : Parameters) (initial : State Word)
    (hw : 0<p.blockColumns) (hsep : p.input ≠ p.output) (hd : InputDefined cfg p initial)
    (a b : State Word)
    (ha : FP.Structural.exec (backend rest cfg u) (source p) initial=some a)
    (hb : FP.Structural.exec (backend rest cfg v) (source p) initial=some b) : a.mem=b.mem := by
  obtain ⟨a',ha',hma⟩ := executionMemory rest cfg u p initial hw hsep hd
  obtain ⟨b',hb',hmb⟩ := executionMemory rest cfg v p initial hw hsep hd
  have ea : a=a' := Option.some.inj (ha.symm.trans ha')
  have eb : b=b' := Option.some.inj (hb.symm.trans hb')
  subst a'
  subst b'
  exact hma.trans hmb.symm

#axiomsClean indexBlock
#axiomsClean indexActive
#axiomsClean offsetGlobal
#axiomsClean quotientGlobal
#axiomsClean correctness
#axiomsClean undefinedPolicyIndependent
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
