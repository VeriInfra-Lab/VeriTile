import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPDecode
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Register-only source execution preserves its arbitrary undef oracle. -/
theorem assignment_step_undef {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (s t : State α) (ha : assignmentOnly st = Bool.true)
    (h : step M st s = some t) : t.undef = s.undef := by
  cases st <;> simp only [assignmentOnly, Bool.false_eq_true] at ha
  case assign d sh name e =>
    cases he : evalExpr M e s with
    | none => simp [step, he] at h
    | some v =>
      simp only [step, he, bind, pure, Option.bind_some, Option.some.injEq] at h
      cases h
      rfl

theorem assignment_run_undef {α : Type} [Inhabited α] (M : Algebra α)
    (code : List ComputeStmt) (s t : State α) (ha : code.all assignmentOnly = Bool.true)
    (h : run M code s = some t) : t.undef = s.undef := by
  induction code generalizing s t with
  | nil =>
    simp only [run, Option.some.injEq] at h
    cases h
    rfl
  | cons st rest ih =>
    simp only [List.all_cons, Bool.and_eq_true] at ha
    cases hu : step M st s with
    | none => simp [run, hu] at h
    | some u =>
      have hr : run M rest u = some t := by
        simpa only [run, hu, bind, Option.bind_some] using h
      exact (ih u t ha.2 hr).trans (assignment_step_undef M st s u ha.1 hu)

theorem decodeExecution_undef (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (decodeExecution rest P p s).val.undef = s.undef :=
  assignment_run_undef (backend rest P) ((source p).surfaceBody.take 28) s _ (by rfl)
    (decodedPrefix rest P p s).property

theorem int32_values_valid {shape : TileShape} (v : TileIndex shape → Int) :
    Int32Bits.valuesValid (fun i => (BitVec.ofInt 32 (v i)).toInt) = Bool.true := by
  simp only [Int32Bits.valuesValid, List.all_eq_true]
  intro i hi
  have hlo := BitVec.le_toInt (BitVec.ofInt 32 (v i))
  have hhi := @BitVec.toInt_lt 32 (BitVec.ofInt 32 (v i))
  simp only [Int32Bits.inRange, decide_eq_true_eq]
  constructor
  · simpa using hlo
  · simpa using hhi

theorem int32_bmod_values_valid {shape : TileShape} (v : TileIndex shape → Int) :
    Int32Bits.valuesValid (fun i => (v i).bmod 4294967296) = Bool.true := by
  simpa only [BitVec.toInt_ofInt] using int32_values_valid v

theorem offsetCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 32 |>.take 1) = offsetCode p := by
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take]
  rfl

@[simp] theorem eval_offsetExpr {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (n : Nat) (s : State α) :
    FP.Structural.evalOp M p (offsetExpr n) s =
      (s.regs .nat [n] "s").map (fun v i => signed16 (signed16 (v i) - 127)) := by
  rw [offsetExpr, FP.Structural.eval_intToSigned, FP.Structural.evalOp]
  rw [FP.Structural.eval_intToSigned]
  simp only [FP.Structural.evalOp]
  cases he : s.regs .nat [n] "s" <;>
    simp only [bind, pure, Option.bind_none, Option.bind_some, Option.map_none, Option.map_some,
      FP.Structural.bop, FP.Structural.numeric, Broadcast.leftIndex, Broadcast.rightIndex, signed16]
  rfl

def signedExponent (n : Nat) : Op .int [n] :=
  Op.intToSigned 32 (.ref .int [n] "s_offset")
def powerCode (p : Parameters) : List ComputeStmt :=
  [.assign .real [p.groups] "s_fp" (.compute (.alg .fp32
    (.libdevicePowi .scalarL (.castFloat .real .real (.const 2)) (signedExponent p.groups))))]
theorem powerCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 33 |>.take 1) = powerCode p := rfl
@[simp] theorem eval_signedExponent {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (n : Nat) (s : State α) :
    FP.Structural.evalOp M p (signedExponent n) s =
      (s.regs .int [n] "s_offset").map (fun v i => (BitVec.ofInt 32 (v i)).toInt) := by
  rw [signedExponent, FP.Structural.eval_intToSigned, FP.Structural.evalOp]
  cases s.regs .int [n] "s_offset" <;> rfl

def scaleCode (p : Parameters) : List ComputeStmt :=
  [.assign .bf16 [p.groups] "s_fp" (.alg (.castFloat .real .bf16
      (.withPrecision .real .fp32 (.ref .real [p.groups] "s_fp")))),
   .assign .int [] "nan_bits" (.alg (intConst 2143289344)),
   .assign .real [] "nan_fp32" (.compute (.bitcast .int32 .fp32 rfl
      (.alg .int32 (.ref .int [] "nan_bits")))),
   .assign .bf16 [] "nan_bf16" (.alg (.castFloat .real .bf16
      (.withPrecision .real .fp32 (.ref .real [] "nan_fp32")))),
   .assign .bf16 [p.groups] "s_fp" (.alg (.where
      (.ne .nat .scalarR (.ref .nat [p.groups] "s") (.constNat 255))
      (.ref .bf16 [p.groups] "s_fp") (.broadcast (.ref .bf16 [] "nan_bf16") [p.groups])))]
theorem scaleCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 34 |>.take 5) = scaleCode p := rfl
#axiomsClean scaleCode_source

#axiomsClean powerCode_source
#axiomsClean eval_signedExponent
#axiomsClean assignment_run_undef
#axiomsClean decodeExecution_undef
#axiomsClean int32_values_valid
#axiomsClean offsetCode_source
#axiomsClean eval_offsetExpr
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
