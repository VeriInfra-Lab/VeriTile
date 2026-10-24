import bench.tritonbench_g.bgmv_expand_slice.Proofs.NativeFP
import VeriTile.Meta.StatementAudit

/-! The complete generic source theorem uses the exact native scalar addition
law at each configuration's actual dtype and compute precision. The former
fp32-rules-only statement is false in the abstract semantics; its checked
countermodel is retained in GeneralCountermodel. The extra native laws are
explicit hypotheses, not new admissions or claims about IEEE hardware. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FPEquivProofs
open VeriTile Triton FP.Structural

abbrev Site := NativeFP.Site
abbrev requirementsFor := NativeFP.requirementsFor

structure Program where
  kernel : ComputeKernel
  domain : List Site

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast : Bool) (output : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (even add : Bool)

def original : Program where
  kernel := TypedSource.original input weight cast output x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add
  domain := requirementsFor (TypedSource.original input weight cast output x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add)

def outputFirst : Program :=
  { original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add with
    kernel := TypedSource.outputFirst input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add }

theorem requirements_present :
    (original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add).domain.length = 1 :=
  NativeFP.requirements_present input weight cast output x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add

/-- Whole-state equivalence, including failures, conditional on the displayed
native scalar laws and finite actual operands. Successful executions are
witnessed separately; the theorem does not assume an output equation. -/
theorem output_first_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α)
    (hc : NativeFP.NativeAdditionCommutes M D
      (original input weight cast output x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel)
    (s : State α)
    (hd : ∀ site ∈ (original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add).domain, site.Holds M D s) :
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (original input weight cast output x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel s =
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (outputFirst input weight cast output x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel s :=
  NativeFP.output_first_execution input weight cast output x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add M D hc s hd
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FPEquivProofs
