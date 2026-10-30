import bench.repairs.tritonbench_g.int8_quantization.Proofs.IOBindings
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.Repairs.Int8Quantization.BiasFold
open VeriTile.Triton FP.Structural Revision002
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Only two reached constant products are simplified. No operand-domain
restriction, associativity or round-to-int agreement is required. -/
def HalfSigns (M : Algebra α) : Prop :=
  M.binary (some .fp32) .real .mul (scalar32 M 0.5) (scalar32 M 1.0) = scalar32 M 0.5 ∧
  M.binary (some .fp32) .real .mul (scalar32 M 0.5) (scalar32 M (-1.0)) = scalar32 M (-0.5)

def segment (p : Parameters) (start count : Nat) : List ComputeStmt :=
  ((biasFoldSource p).surfaceBody.drop start).take count

theorem unchanged_prefix (p : Parameters) : segment p 0 16 = Revision002.segment p 0 16 := rfl

theorem registers_run (M : Algebra α) (signs : HalfSigns M)
    (ops : Model.Primitives α) (p : Parameters) (positive : 0<p.width) (entry : State α) :
    run (Model.backend M ops) (segment p 16 5) (reduced M ops p positive entry) =
      some (registers M ops p positive entry) := by
  have choice (test : Bool) :
      M.binary (some .fp32) .real .mul (scalar32 M 0.5)
        (if test then scalar32 M 1.0 else scalar32 M (-1.0)) =
          if test then scalar32 M 0.5 else scalar32 M (-0.5) := by
    cases test
    · exact signs.2
    · exact signs.1
  simp [segment,biasFoldSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Model.backend,Option.bind,numericLt,numericLe,bop,numeric,ofFloat,toFloat,
    registers,reduced,classified,quotient,normalized,clipped,biased,quantized,ComputeDType.eraseDType,choice]
  all_goals
    delta bop Model.offset Model.group loaded transformed flags sanitized scalar32 maximum scale usable denominator quotient normalized clipped biased quantized
    first | rfl | simp_all (config := {maxSteps:=1000000}) only [Bool.false_eq_true, ↓reduceIte]
    all_goals try rfl



theorem early_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0<p.width) (entry : State α) :
    run (Model.backend M ops) (segment p 0 16) entry = some (reduced M ops p positive entry) := by
  rw [unchanged_prefix]
  have parts : Revision002.segment p 0 16 = Revision002.segment p 0 8 ++
    Revision002.segment p 8 1 ++ Revision002.segment p 9 3 ++ Revision002.segment p 12 4 := rfl
  rw [parts]
  simp only [run_append,addressed_run,prescaled_run,classified_run M ops p positive,reduced_run,Option.bind_some]

theorem source_run (M : Algebra α) (signs : HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (entry : State α) :
    FP.Structural.exec (Model.backend M ops) (biasFoldSource p) entry =
      some (finalState M ops p positive entry) := by
  have parts : (biasFoldSource p).surfaceBody = segment p 0 16 ++ segment p 16 5 ++
    Revision002.segment p 21 2 := rfl
  change run (Model.backend M ops) (biasFoldSource p).surfaceBody entry = _
  rw [parts]
  simp only [run_append,early_run M ops p positive,registers_run M signs,
    stores_run,Option.bind_some]

theorem middle_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (Model.backend M ops) bounds (segment p 8 13) s := by
  apply SourceSafety.trace_from_steps
  intro st member t
  simp only [segment,biasFoldSource,ComputeKernel.surfaceBody,List.drop,List.take,
    List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    cases query : p.query <;>
      simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
        Safety.traceSafeList_unfold,step,evalOp_unfold,evalExpr,Option.bind,query]
  all_goals split <;> trivial

theorem source_safe (M : Algebra α) (signs : HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i, Model.active p s i → Model.offset p s i < bounds xRegion)
    (output : ∀ i, Model.active p s i → Model.offset p s i < bounds outRegion)
    (scale : Model.group p s < bounds scaleRegion) :
    TraceSafe (Model.backend M ops) bounds (biasFoldSource p) s := by
  have parts : (biasFoldSource p).surfaceBody = Revision002.segment p 0 8 ++
    segment p 8 13 ++ Revision002.segment p 21 2 := rfl
  have middle : run (Model.backend M ops) (segment p 8 13) (addressed M p s) =
      some (registers M ops p positive s) := by
    have parts : segment p 8 13 = Revision002.segment p 8 1 ++ Revision002.segment p 9 3 ++
      Revision002.segment p 12 4 ++ segment p 16 5 := rfl
    rw [parts]
    simp only [run_append,prescaled_run,classified_run M ops p positive,reduced_run,
      registers_run M signs,Option.bind_some]
  unfold TraceSafe
  rw [parts,List.append_assoc]
  apply (Safety.traceSafeList_append _ _ _ _ _).mpr
  constructor
  · exact SourceSafety.prefix_safe M ops p s bounds input
  · intro t ht
    have eq : t = addressed M p s := Option.some.inj (ht.symm.trans (addressed_run M ops p s))
    subst t
    apply (Safety.traceSafeList_append _ _ _ _ _).mpr
    constructor
    · exact middle_safe M ops p _ bounds
    · intro t ht
      have eq : t = registers M ops p positive s := Option.some.inj (ht.symm.trans middle)
      subst t
      exact SourceSafety.stores_safe M ops p positive s bounds output scale

/-- A fixed software FP32 multiplication model with exact source constants;
other numerical instructions remain explicit primitive ports. -/
def foldedLiteral (rest : Algebra (FP.Value .fp32)) (p : Option ComputeDType)
    (d : FloatDType) (x : ℝ) : FP.Value .fp32 := by
  classical
  exact if x = (1/2 : ℝ) then (FP.Value.ofBits .fp32 0x3f000000)
    else if x = (-1 : ℝ) then (FP.Value.ofBits .fp32 0xbf800000)
    else if x = (-1/2 : ℝ) then (FP.Value.ofBits .fp32 0xbf000000)
    else rest.literal p d x

def model (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  FP.Software.scalarFP32 { rest with literal := foldedLiteral rest }

theorem half_times_one : FP.mul .ieeeValue (FP.Value.ofBits .fp32 0x3f000000)
    (FP.Value.ofBits .fp32 0x3f800000) = FP.Value.ofBits .fp32 0x3f000000 := by decide +kernel

theorem half_times_neg_one : FP.mul .ieeeValue (FP.Value.ofBits .fp32 0x3f000000)
    (FP.Value.ofBits .fp32 0xbf800000) = FP.Value.ofBits .fp32 0xbf000000 := by decide +kernel

theorem model_signs (rest : Algebra (FP.Value .fp32)) : HalfSigns (model rest) := by
  constructor <;>
    norm_num [HalfSigns,model,foldedLiteral,scalar32,FP.Software.scalarFP32,FP.Software.fp32Context]
  · exact half_times_one
  · exact half_times_neg_one

#axiomsClean registers_run
#axiomsClean source_run
#axiomsClean source_safe
#axiomsClean model_signs
end
end VeriTile.Bench.Repairs.Int8Quantization.BiasFold
