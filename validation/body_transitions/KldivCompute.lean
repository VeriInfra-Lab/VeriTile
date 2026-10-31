import bench.tritonbench_g.kldiv_compute.Kernels
import bench.tritonbench_g.kldiv_compute.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

/-! Transition evidence: the new canonical raw quotation equals every
previously admitted typed-source profile. The pre-existing native support
classifier excludes both-fp8 input pairs; no supported profile is removed.
This file is archived in Git before the legacy generator is retired. -/
namespace VeriTile.Bench.Repairs.KldivCompute.SourceTransition
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource

/-- Freeze the former all-FP32 macro expansion too: its selected alias
will otherwise follow the canonical entry during this transition. -/
def legacyFamily (xType yType outType : Storage) (version : Bool) : Source :=
  match xType,yType,outType,version with
  | .fp32,.fp32,.fp32,Bool.false => kldivSource%[tl.float32,tl.float32,tl.float32,original]
  | .fp32,.fp32,.fp32,Bool.true => kldivSource%[tl.float32,tl.float32,tl.float32,commuted]
  | _,_,_,_ => kernel xType yType outType version

theorem all_supported_profiles (xType yType outType : Storage) (version : Bool)
    (X Y O : RegionName) (N B : Nat)
    (supported : sourceSupported xType yType = Bool.true) :
    KernelDSL.family xType yType outType version X Y O N B =
      legacyFamily xType yType outType version X Y O N B := by
  cases xType <;> cases yType <;> cases outType <;> cases version
  all_goals simp only [sourceSupported,Bool.false_eq_true] at supported
  all_goals rfl

#axiomsClean all_supported_profiles
end VeriTile.Bench.Repairs.KldivCompute.SourceTransition
