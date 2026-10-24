import bench.tritonbench_g.iv_dependent_matmul.Proofs.Real

/-! Auxiliary conditional exact-arithmetic domain for the selected FP32 graph.
These equations are generally false for arbitrary rounded FP32/native TF32.
They are not pending hardware conformance obligations and are not imported by
the public correctness or FP equivalence contracts. The scalar
outer add is separate from Dot. DotAcc has its own primitive obligation. Terminal
FP32→FP16 narrowing and FP16→FP32 widening are separate, with actual half bits
rather than an FP32 reinterpretation. None of these obligations concerns program
success, indexing, memory, a whole-kernel output, or compiler graph fusion. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Numerical
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole RealTarget
noncomputable section

abbrev halfBits := TritonBenchG.QuantizeKvTransform.MixedBits.half

def halfReal (v : Word) : ℝ :=
  match (halfBits v).toRat? with
  | some q => (q.num : ℝ)/(q.den : ℝ)
  | none => 0

def dotValue [Inhabited Word] (D : Dot) (p : Parameters) (s : State Word) (k : Nat) : Accumulator p :=
  D.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch := []) (aValues p s k) (bValues p s k)

def exactBlock [Inhabited Word] (p : Parameters) (s : State Word) (k : Nat) (q : TileIndex [p.blockRows,p.blockColumns]) : ℝ :=
  ∑ j : Fin p.blockInner,
    realValue (aValues p s k (q.1,j,PUnit.unit))*realValue (bValues p s k (j,q.2.1,PUnit.unit))

theorem zero_real : realValue (FP.Value.zero .fp32)=0 := by
  have h : (FP.Value.zero .fp32).toRat?=some 0 := by decide +kernel
  simp [realValue,h]

/-- Domain and laws only at actual reached primitive operands. `finiteHalf`
excludes terminal overflow from this finite-Real interpretation explicitly;
FP value and schedule-equivalence theorems impose no such numerical restriction.
`roundHalf` denotes the selected binary16 rounding interpretation, not identity. -/
structure LocalNumerics [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s : State Word) (roundHalf : ℝ → ℝ) : Prop where
  finiteReached : ∀ pid k a, Reached cfg D F g r.parameters (instanceState pid s) k a →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true → (a q).toRat?.isSome=Bool.true
  dot : g=.source → ∀ pid k a, Reached cfg D F g r.parameters (instanceState pid s) k a →
    k<count r.parameters → ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      realValue (dotValue D r.parameters (instanceState pid s) k q)=
        exactBlock r.parameters (instanceState pid s) k q
  add : g=.source → ∀ pid k a, Reached cfg D F g r.parameters (instanceState pid s) k a →
    k<count r.parameters → ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      realValue (FP.add cfg (a q) (dotValue D r.parameters (instanceState pid s) k q))=
        realValue (a q)+realValue (dotValue D r.parameters (instanceState pid s) k q)
  dotAcc : g=.accumulator → ∀ pid k a, Reached cfg D F g r.parameters (instanceState pid s) k a →
    k<count r.parameters → ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      realValue (F.apply (some .fp32) .fp32 .fp32 .real {} (batch := [])
        (aValues r.parameters (instanceState pid s) k) (bValues r.parameters (instanceState pid s) k) a q)=
        realValue (a q)+exactBlock r.parameters (instanceState pid s) k q
  finiteHalf : ∀ pid a, Reached cfg D F g r.parameters (instanceState pid s) (count r.parameters) a →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      (halfBits (castWord cfg .single .half (a q))).toRat?.isSome=Bool.true
  narrow : ∀ pid a, Reached cfg D F g r.parameters (instanceState pid s) (count r.parameters) a →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      halfReal (castWord cfg .single .half (a q))=roundHalf (realValue (a q))
  widen : ∀ pid a, Reached cfg D F g r.parameters (instanceState pid s) (count r.parameters) a →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      realValue (castWord cfg .half .single (castWord cfg .single .half (a q)))=
        halfReal (castWord cfg .single .half (a q))

theorem to_boundary [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s : State Word) (roundHalf : ℝ → ℝ)
    (law : LocalNumerics cfg D F g r s roundHalf) :
    NumericalBoundary cfg D F g r s roundHalf where
  zero := zero_real
  finite := law.finiteReached
  step := by
    intro pid k a hr hk q hq
    cases g with
    | source =>
      change realValue (FP.add cfg (a q) (dotValue D r.parameters (instanceState pid s) k q))=_
      rw [law.add rfl pid k a hr hk q hq,law.dot rfl pid k a hr hk q hq]
      rfl
    | accumulator => exact law.dotAcc rfl pid k a hr hk q hq
  narrowing := by
    intro pid a hr q hq
    exact (law.widen pid a hr q hq).trans (law.narrow pid a hr q hq)

#axiomsClean zero_real
#axiomsClean to_boundary
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Numerical
