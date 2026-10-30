import bench.tritonbench_g.matmul_leakyrelu.PairIO
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourcePair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

/-- Only a local K32 FP32 instruction fragment is registered. No complete
kernel, activation, memory, grid or output relation occurs in the assumption. -/
def fragmentRequest : Whole.Request where
  A := "A"; B := "B"; C := "C"
  rows := 32; inner := 32; innerB := 32; columns := 32
  strideAM := 32; strideAK := 1; strideBK := 32; strideBN := 1
  aContiguous := Bool.true; activation := ""

def separateFragment : List ComputeStmt :=
  ((FPValue.body fragmentRequest.parameters .single).take 3).drop 2
def contractedFragment : List ComputeStmt :=
  ((FPFused.body fragmentRequest.contractedParameters .single).take 3).drop 2

/-- Pending numerical validation; this record does not claim either gate ran.
The shapes are the local source instruction tile, independent of matrix sizes. -/
def contractionEntry : Spec.RuleEntry ComputeStmt where
  rule := {
    lhs := separateFragment
    rhs := contractedFragment
    contract := {
      ruleID := "MATMUL-LEAKYRELU-FP32-K32-DOT-ADD-TO-DOTACC"
      instanceKey := "matmul_leakyrelu.fp32.tf32.k32.live_acc.all_words.v1"
      configuration := Lean.Json.mkObj [
        ("precision", "fp32"), ("dot_input_precision", "tf32"),
        ("tile", Lean.toJson ([32,32,32] : List Nat)),
        ("domain", "all FP32 payloads; padded zero lanes; arbitrary live FP32 accumulator"),
        ("backend", "SM90/Triton 3.7.1 selected profile"),
        ("status", "deferred; no experimental gates run")]
      warningPolicy := .passOnly
      description := "Local separate FP32 dot, fp32-to-real carrier cast and add to seeded FP32 dotAcc; numerical assumption pending validation, not IEEE word equality." } }
  evidence := { instanceKey := "matmul_leakyrelu.fp32.tf32.k32.live_acc.all_words.v1" }

/-- Evidence remains an explicit conditional parameter. The pending record
above has no fabricated acceptance proof; an actual future record must discharge
its own gates and provenance before constructing this value. -/
structure Rules where
  evidence : Spec.Evidence
  accepted : Spec.AcceptedAtom (⟨contractionEntry.rule,evidence⟩ : Spec.RuleEntry ComputeStmt)

def Rules.entry (R : Rules) : Spec.RuleEntry ComputeStmt := ⟨contractionEntry.rule,R.evidence⟩
def Rules.assumptions (R : Rules) : Spec.Assumptions ComputeStmt := [R.entry]
instance : CoeOut Rules (Spec.Assumptions ComputeStmt) := ⟨Rules.assumptions⟩

/-- This actual local atom is interpreted to obtain the dot contraction law
used in the complete source proof. It is not an unused audit witness. -/
theorem admitted (R : Rules) :
    Spec.Derivation R.assumptions separateFragment contractedFragment :=
  Spec.Derivation.atom R.entry (by simp [Rules.assumptions]) R.accepted

#print_fp_assumptions admitted

/-- Interpretation of the registered local typed instruction rewrite. The
model says nothing about successful kernels, safety, frames or final outputs. -/
def Models (R : Spec.Assumptions ComputeStmt) (M : Algebra Pair.Word)
    (D : FPValue.Dot Pair.Word) (F : FPFused.DotAcc Pair.Word) : Prop :=
  Spec.Derivation R separateFragment contractedFragment → Pair.DotAddContract M D F

local instance : Inhabited Pair.Word := ⟨FP.Value.zero .fp32⟩

def IOEquivalent {p : KernelInterface.Parameters} {order : Schedule p.request}
    (R : Spec.Assumptions ComputeStmt) (left right : IOProgram p order) : Prop :=
  ∀ (cfg : FP.Config) (rest : Algebra Pair.Word) (D : FPValue.Dot Pair.Word)
    (F : FPFused.DotAcc Pair.Word) (Q : Whole.Numerics Pair.Word),
    Whole.PrimitiveLaws (Pair.fp32Algebra cfg rest) D Q → F.le = D.le →
    Models R (Pair.fp32Algebra cfg rest) D F →
    (programIO cfg rest D F left).Equivalent (variant cfg rest D F left right)

instance {p : KernelInterface.Parameters} {order : Schedule p.request} :
    Spec.ProgramSyntax (IOProgram p order) where
  Statement := ComputeStmt
  Signature := Unit
  signature := fun _ => ()
  body := fun program => program.kernel.surfaceBody
  structural := some (fun _ _ => False)
  numerical := IOEquivalent
  sameContext := Eq

theorem io_equivalent (R : Rules) (p : KernelInterface.Parameters) (order : Schedule p.request)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) :
    IOEquivalent R.assumptions (original p order) (contracted p order) := by
  intro cfg rest D F Q ports comparison model
  have localContract := model (admitted R)
  exact Pair.equivalent (Pair.fp32Algebra cfg rest) D F Q ports localContract comparison
    (Pair.fp32_mul_commutes cfg rest) p order separateA separateB

#axiomsClean io_equivalent
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourcePair
