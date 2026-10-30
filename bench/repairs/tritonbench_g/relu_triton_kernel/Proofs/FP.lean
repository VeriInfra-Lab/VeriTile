import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Host
import bench.tritonbench_g.relu_triton_kernel.Proofs.SoftwareFP32

namespace VeriTile.Bench.Repairs.ReluTritonKernel.FPProofs
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluTritonKernel.SoftwareFP32
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent all-word classification. NaNs and negative nonzero values
become +0; both zero signs and every nonnegative ordered value are retained. -/
def reference (x : Word) : Word :=
  if x.isNaN || (x.sign && !x.isZero) then FP.Value.zero .fp32 else x

def backend (rest : Algebra Word) : Algebra Word :=
  {rest with
    fp32Load:=id
    literal:=fun p d x=>if p=some .fp32 ∧ d=.real ∧ x=0 then FP.Value.zero .fp32 else rest.literal p d x
    compareLe:=fun p d=>if p=some .fp32 ∧ d=.real then some orderedLe else rest.compareLe p d}

theorem compare (rest : Algebra Word) :
    (backend rest).compareLe (some .fp32) .real=some orderedLe := by simp [backend]

theorem classification (x : Word) : wordRelu x=reference x := by
  have zn : (FP.Value.zero .fp32).isNaN=Bool.false := by decide
  have zz : (FP.Value.zero .fp32).isZero=Bool.true := by decide
  have zs : (FP.Value.zero .fp32).sign=Bool.false := by decide
  have zb : (FP.Value.zero .fp32).bits.toNat=0 := by decide
  cases hn : x.isNaN <;> cases hz : x.isZero <;> cases hs : x.sign <;>
    simp [wordRelu,orderedLe,reference,hn,hz,hs,zn,zz,zs,zb]

theorem scalar (rest : Algebra Word) (x : Word) :
    Structural.scalar (backend rest) orderedLe x=reference x := by
  simpa [Structural.scalar,backend,wordRelu] using classification x

/-- Ordinary local literal/comparison interpretation for a backend. Raw where-selection and same-type load/store are structural semantics.
This does not assume the desired tensor result or restrict admitted values. -/
structure PrimitiveInterpretation (M : Algebra Word) : Prop where
  positiveZero : M.literal (some .fp32) .real 0=FP.Value.zero .fp32
  orderedComparison : M.compareLe (some .fp32) .real=some orderedLe

theorem scalar_if_local (M : Algebra Word) (h : PrimitiveInterpretation M) (x : Word) :
    Structural.scalar M orderedLe x=reference x := by
  simpa [Structural.scalar,h.positiveZero,wordRelu] using classification x

theorem special_words :
    (reference (FP.Value.ofBits .fp32 0x80000000)).bits=0x80000000 ∧
    (reference (FP.Value.ofBits .fp32 0x7fc12345)).bits=0 ∧
    (reference (FP.Value.ofBits .fp32 0x7f800001)).bits=0 ∧
    (reference (FP.Value.ofBits .fp32 0x7f800000)).bits=0x7f800000 ∧
    (reference (FP.Value.ofBits .fp32 0xff800000)).bits=0 ∧
    (reference (FP.Value.ofBits .fp32 1)).bits=1 ∧
    (reference (FP.Value.ofBits .fp32 0x80000001)).bits=0 := by decide

#specNonCircular reference avoiding [kernel, Coordinates.kernel,Host.runWrapper,Structural.scalar,wordRelu]
#axiomsClean classification
#axiomsClean scalar_if_local
#axiomsClean special_words
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.FPProofs
