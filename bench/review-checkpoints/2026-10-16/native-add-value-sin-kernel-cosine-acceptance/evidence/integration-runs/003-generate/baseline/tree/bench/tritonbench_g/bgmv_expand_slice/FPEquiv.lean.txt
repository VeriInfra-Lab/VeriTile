import bench.tritonbench_g.bgmv_expand_slice.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Full-source output-first equivalence for every input, weight, cast and
output dtype. The native scalar addition law and finite actual operand domain
are explicit. Existing fp32 experiment rules discharge the fp32 profile;
native half/bfloat16 laws remain additional backend premises. This corrects
the former unsound fp32-only contract without restricting source choices. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.Equivalence
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec

abbrev original := FPEquivProofs.original
abbrev outputFirst := FPEquivProofs.outputFirst

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (even add : Bool)

specification bgmv_output_first_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α)
    (native_add : NativeFP.NativeAdditionCommutes M D
      (original input weight cast out_dtype x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel)
    (s : State α)
    (finite_operands : ∀ site ∈ (original input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add).domain, site.Holds M D s) :
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (original input weight cast out_dtype x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel s =
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (outputFirst input weight cast out_dtype x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel s := by
  exact FPEquivProofs.output_first_equiv input weight cast out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add M D native_add s finite_operands
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.Equivalence
