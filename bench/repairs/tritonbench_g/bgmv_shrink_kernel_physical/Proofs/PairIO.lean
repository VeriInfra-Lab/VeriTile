import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.PairIO
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PairConnection
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOImplements
import VeriTile.Triton.Float.PerAddressProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PairIO
open VeriTile Triton FP.Structural FPValues PairConnection
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 24000
variable {A : Type} [Inhabited A]

theorem machine_comm (M : Algebra A) (P : Primitives A) (comm : MultiplicationCommutes M) :
    MultiplicationCommutes (machine M P) := by
  simpa [MultiplicationCommutes,machine,Algebra.withDefaultPrecision,resolvePrecision] using comm

theorem invocation_run (M : Algebra A) (P : Primitives A) (h : Call) (scale : A)
    (comm : MultiplicationCommutes M) (p : Program h) (s : State A) :
    (IO.invocation M P h scale p).run s = (invocation M P h scale p).run s := by
  exact (source_pair (machine M P) (machine_comm M P comm) h).run _

theorem invocation_safe (M : Algebra A) (P : Primitives A) (h : Call) (scale : A)
    (comm : MultiplicationCommutes M) (p : Program h) (bounds : RegionName → Nat) (s : State A) :
    (IO.invocation M P h scale p).Safe bounds s ↔ (invocation M P h scale p).Safe bounds s := by
  exact (source_pair (machine M P) (machine_comm M P comm) h).safe bounds _

/-- Existing source-derived ordered target, used only to reuse its complete
execution/safety/locality/coverage certificate in the pair transport. -/
def formula (M : Algebra A) (P : Primitives A) (h : Call)
    (indices : Nat → Int) (scale : A) (orders : Orders h)
    (vals : (IO.programIO M P h indices scale orders id).Inputs) :
    (IO.programIO M P h indices scale orders id).Outputs :=
  Fin.cases (fun j => ofFloat h.outputType
    (FPValues.target M P h orders (IOValues.data h indices scale vals) j.1 j.2.1))
    (fun i => Fin.elim0 i)

/-- Every complete batch × 64 CTA and each output address's independent
atomic order are retained. Success, access safety, tagged output equality,
source locality/coverage and both exterior frames are all conclusions. -/
theorem equivalent (M : Algebra A) (P : Primitives A) (h : Call)
    (indices : Nat → Int) (scale : A) (orders : Orders h) (comm : MultiplicationCommutes M) :
    (IO.programIO M P h indices scale orders id).EquivalentOnCells
      (candidate M P h indices scale orders) (IO.domain h indices) := by
  apply PerAddressProgramIO.EquivalentOnCells.of_run_safe
    (candidate M P h indices scale orders) rfl
    (fun _ p s => invocation_run M P h scale comm p s)
    (fun _ p bounds s => invocation_safe M P h scale comm p bounds s)
    (f := formula M P h indices scale orders)
  apply IOImplements.implements
  intro vals b n
  rfl

/-- Fix precisely the rewritten FP32 operation to the existing strict software
multiplication model; every other local operation remains shared unchanged. -/
def strictAlgebra (M : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  { M with binary := fun p d op a b =>
    if p=some .fp32 ∧ d=.real ∧ op=.mul then FP.mul cfg a b else M.binary p d op a b }

theorem strict_multiplication (M : Algebra (FP.Value .fp32)) (cfg : FP.Config) :
    MultiplicationCommutes (strictAlgebra M cfg) := by
  intro a b
  simpa [strictAlgebra] using FP.ScalarOpsLaws.mul_comm cfg a b

#axiomsClean equivalent
#axiomsClean strict_multiplication
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PairIO
