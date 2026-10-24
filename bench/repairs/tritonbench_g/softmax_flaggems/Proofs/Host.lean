import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Grid

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Host
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open HostGrid
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Canonical real logical projection. Explicit repaired load defaults, rather
than a claim about unspecified native values, justify the old zero-sampler
transport. All initial registers and memory cells are retained. -/
def entry (s : BlockState) : BlockState := { s with undef := fun _ _ => 0 }

theorem forward_undef (Y X : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s t : BlockState) (he : forwardRun .fp32 .fp32 Y X d c s = some t) :
    t.undef = s.undef := by
  unfold forwardRun at he
  split at he
  · apply Grid.run_undef _ _ _ _ _ s t he
    cases h : c.one
    · exact RealForwardNonInner.original_flattenOk .fp32 .fp32 Y X d.M d.N d.K c.B c.C
    · exact RealForwardNonInner.oneTile_original_flattenOk .fp32 .fp32 Y X d.M d.N d.K c.B c.C
  · apply Grid.run_undef _ _ _ _ _ s t he
    cases h : c.one
    · exact RealForwardInner.original_flattenOk .fp32 .fp32 Y X d.M d.N c.B
    · exact RealForwardInnerOneTile.original_flattenOk .fp32 .fp32 Y X d.M d.N c.B

noncomputable def nonempty (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) : Option AutogradResult := do
  let d := dimensions shape ax
  let pre := prepareForward initial X P shape.prod inputValues emptyOutput
  let mid ← forwardRun .fp32 .fp32 P X d (forwardConfig waves d sms) pre
  let beforeBackward := prepareBackward mid P G Y shape.prod gradValues emptyGradient
  let final ← Grid.tunedBackward P G Y d trials choice beforeBackward
  let out := forwardMetadata layouts inputStrides P shape .fp32 device
  return ⟨mid, final, out, ⟨out, ax.val⟩,
    gradientMetadata layouts inputStrides Y shape .fp32 device, none, none⟩

theorem nonempty_eq (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (X P G Y : RegionName) (initial : BlockState) (hs : ZeroPadding.Zero initial)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) :
    autograd shape ax .fp32 none device sms waves layouts inputStrides trials choice X P G Y
      initial inputValues emptyOutput gradValues emptyGradient =
    nonempty shape ax device sms waves layouts inputStrides trials choice X P G Y
      initial inputValues emptyOutput gradValues emptyGradient := by
  let d := dimensions shape ax
  let pre := prepareForward initial X P shape.prod inputValues emptyOutput
  have hp : ZeroPadding.Zero pre := hs
  cases hf : forwardRun .fp32 .fp32 P X d (forwardConfig waves d sms) pre with
  | none => simp [autograd, nonempty, outputStorage, d, pre] at hf ⊢; simp_all
  | some mid =>
    have hm : ZeroPadding.Zero mid := (forward_undef P X d _ pre mid hf).trans hp
    have hb : ZeroPadding.Zero (prepareBackward mid P G Y shape.prod gradValues emptyGradient) := by
      by_cases hg : G = P <;> simpa [ZeroPadding.Zero, prepareBackward, writeSlice, hg] using hm
    have ht := Grid.tuned_eq P G Y d trials choice _ hb
    simp only [autograd, nonempty, outputStorage, Option.getD_none]
    change ((forwardRun .fp32 .fp32 P X d (forwardConfig waves d sms) pre).bind _) = _
    rw [hf]
    simp only [bind, Option.bind]
    rw [ht]

/-- Independent conventional VJP, using actual saved output and upstream reads.
It contains no source execution, kernel result, padding sample, or tile length. -/
noncomputable def logicalBackward (s : BlockState) (P G : RegionName)
    (d : Dimensions) (a : MemCellAddr) : ℝ :=
  if 1 < d.K then
    RealBackwardNonInner.unpaddedExpected s P G (a.2 / d.K / d.N) d.N d.K
      (a.2 / d.K % d.N) (a.2 % d.K)
  else
    RealBackwardInner.unpaddedExpected s P G d.M d.N (a.2 / d.N) (a.2 % d.N)

theorem backward_standard (s : BlockState) (hs : ZeroPadding.Zero s)
    (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0 < d.N) (hK : 0 < d.K) (hB : 0 < c.B)
    (hone : c.one = oneTile d.N c.B) (o : Nat) (ho : o < d.M * d.N * d.K) :
    backwardValue s P G d c (Y, o) = logicalBackward s P G d (Y, o) := by
  have hpad : d.N ≤ RealBackwardInner.paddedLength c.one d.N c.B := by
    rw [hone]
    simp only [RealBackwardInner.paddedLength, oneTile]
    split_ifs with h
    · simpa using h
    · exact RealAddress.coverage hB
  by_cases hk : 1 < d.K
  · have coords := coordinate_noninner hN hK ho
    simpa [backwardValue, logicalBackward, hk, backwardNonInnerValue] using
      RealBackwardNonInner.expected_eq_of_padding s P G (o / d.K / d.N) d.N d.K
        (RealBackwardNonInner.paddedLength c.one d.N c.B) (o / d.K % d.N) (o % d.K)
        hpad coords.2.1 coords.2.2.1 (RealBackwardNonInner.zero_undef_padding s P G _ _ _ _ _ hs)
  · have hk1 : d.K = 1 := by omega
    have ho' : o < d.M * d.N := by simpa [hk1] using ho
    have coords := coordinate_inner hN ho'
    simpa [backwardValue, logicalBackward, hk, backwardInnerValue] using
      RealBackwardInner.expected_eq_of_padding s P G d.M d.N
        (RealBackwardInner.paddedLength c.one d.N c.B) (o / d.N) (o % d.N)
        hpad coords.2.1 coords.1 (RealBackwardInner.zero_undef_padding s P G _ _ _ _ hs)

def emptyResult (shape : List Nat) (ax : Fin shape.length)
    (device : Nat) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (P Y : RegionName) (initial : BlockState) : AutogradResult :=
  let out := forwardMetadata layouts inputStrides P shape .fp32 device
  ⟨initial, initial, out, ⟨out, ax.val⟩,
    gradientMetadata layouts inputStrides Y shape .fp32 device, none, none⟩

/-- The empty branch precedes dimension division and every launch, in both
forward and backward. Invalid dim remains rejected before that branch. -/
noncomputable def invoke (shape : List Nat) (dim : Int)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) : Except HostError AutogradResult :=
  match normalizeDim shape dim with
  | none => .error .invalidDim
  | some ax =>
    if shape.prod = 0 then .ok (emptyResult shape ax device layouts inputStrides P Y (entry initial))
    else match nonempty shape ax device sms waves layouts inputStrides trials choice X P G Y
      (entry initial) inputValues emptyOutput gradValues emptyGradient with
      | none => .error .kernelFailure
      | some result => .ok result

def ResultSpec (shape : List Nat) (ax : Fin shape.length)
    (device : Nat) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) (result : AutogradResult) : Prop :=
  let pre := prepareForward (entry initial) X P shape.prod inputValues emptyOutput
  let before := prepareBackward result.forwardState P G Y shape.prod gradValues emptyGradient
  FlatResult P shape.prod (forwardValue pre X (dimensions shape ax)) pre result.forwardState ∧
  FlatResult Y shape.prod (logicalBackward before P G (dimensions shape ax)) before result.finalState ∧
  (∀ o, before.readMem P o = result.forwardState.readMem P o) ∧
  (∀ o, result.finalState.readMem P o = result.forwardState.readMem P o) ∧
  result.output = forwardMetadata layouts inputStrides P shape .fp32 device ∧
  result.saved = ⟨result.output, ax.val⟩ ∧
  result.gradient = gradientMetadata layouts inputStrides Y shape .fp32 device ∧
  result.dimGradient = none ∧ result.dtypeGradient = none ∧
  BlockState.WriteWithin (hostFootprint X P G Y shape.prod) initial result.finalState

theorem empty_spec (shape : List Nat) (ax : Fin shape.length) (hz : shape.prod = 0)
    (device : Nat) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) :
    ResultSpec shape ax device layouts inputStrides X P G Y initial
      inputValues emptyOutput gradValues emptyGradient
      (emptyResult shape ax device layouts inputStrides P Y (entry initial)) := by
  simp [ResultSpec, emptyResult, hz, prepareForward, prepareBackward, writeSlice,
    FlatResult, BlockState.WriteWithin, BlockState.AgreesOutside, fullSlice, hostFootprint, entry]

theorem whole_call_real (shape : List Nat) (dim : Int) (ax : Fin shape.length)
    (hnorm : normalizeDim shape dim = some ax)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ)
    (hXP : X ≠ P) (hPY : P ≠ Y) (hGY : G ≠ Y) :
    ∃ result,
      invoke shape dim device sms waves layouts inputStrides trials choice X P G Y initial
        inputValues emptyOutput gradValues emptyGradient = .ok result ∧
      ResultSpec shape ax device layouts inputStrides X P G Y initial
        inputValues emptyOutput gradValues emptyGradient result := by
  by_cases hz : shape.prod = 0
  · exact ⟨_, by simp [invoke, hnorm, hz],
      empty_spec shape ax hz device layouts inputStrides X P G Y initial
        inputValues emptyOutput gradValues emptyGradient⟩
  · have hprod : 0 < shape.prod := Nat.pos_of_ne_zero hz
    rw [shape_factorization shape ax] at hprod
    have hm := Nat.pos_of_mul_pos_right (Nat.pos_of_mul_pos_right hprod)
    have hn := Nat.pos_of_mul_pos_left (Nat.pos_of_mul_pos_right hprod)
    have hk : 0 < (dimensions shape ax).K := by
      rw [dimensions_suffix shape ax hm hn]
      exact Nat.pos_of_mul_pos_left hprod
    obtain ⟨result, he, hf, hb, hsaved, hsavedFinal, hout, hctx, hgrad, hd, hdt, hframe⟩ :=
      softmax_autograd_host_grid_real shape ax .fp32 none device sms waves layouts inputStrides
        trials choice X P G Y (entry initial) inputValues emptyOutput gradValues emptyGradient
        hm hn hk hXP hPY hGY
    have hnew := (nonempty_eq shape ax device sms waves layouts inputStrides trials choice
      X P G Y (entry initial) rfl inputValues emptyOutput gradValues emptyGradient).symm.trans he
    refine ⟨result, by simp [invoke, hnorm, hz, hnew], hf, ?_, hsaved, hsavedFinal,
      hout, hctx, hgrad, hd, hdt, hframe⟩
    refine ⟨hb.1, ?_, hb.2.2⟩
    intro o ho
    have hs : ZeroPadding.Zero (prepareBackward result.forwardState P G Y shape.prod gradValues emptyGradient) := by
      have hmid : ZeroPadding.Zero result.forwardState := hf.2.2
      by_cases hg : G = P <;> simpa [ZeroPadding.Zero, prepareBackward, writeSlice, hg] using hmid
    exact (hb.2.1 o ho).trans (backward_standard _ hs P G Y (dimensions shape ax)
      (backwardConfig (dimensions shape ax) choice) hn hk
      (backwardConfig_positive (dimensions shape ax) choice).1
      (backwardConfig_one (dimensions shape ax) choice) o
      (by simpa [dimensions_numel shape ax hm hn] using ho))

#axiomsClean whole_call_real
#axiomsClean backward_standard
#axiomsClean empty_spec
#specNonCircular logicalBackward avoiding [invoke, nonempty, Grid.backwardRun, Grid.trialRun, Grid.tunedBackward, run, exec]
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Host
