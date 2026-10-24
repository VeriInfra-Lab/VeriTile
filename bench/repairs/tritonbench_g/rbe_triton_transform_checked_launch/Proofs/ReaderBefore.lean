import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.IOCorrectness

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.ReaderBefore
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.ProgramIO
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section

/-- FP32 source, complete two-axis grid and contiguous input/output windows.
The shared public descriptor in IO.lean lists every channel, mask and ABI field. -/
@[reducible] def realIO (r : Request) (X Y : RegionName) (qs : List Runtime.Program) :
    ProgramIO ℝ ℝ := rotationIO ideal r X Y qs id id

/-- Independent dense rotation, including zero for a missing odd partner. -/
def result (r : Request) (xs : TileIndex [r.size] → ℝ) (j : TileIndex [r.size]) : ℝ :=
  let (b,m,c) := coordinates r j.1.val
  let n := c/2*2
  let x := logicalInput r xs
  let u := x (r.offset b m n)
  let v := if n+1<r.columns then x (r.offset b m (n+1)) else 0
  let a := ((r.position : ℝ)+(m : ℝ))/Real.rpow 10000 ((n%128 : Nat)/(128 : ℝ))
  if decide (c%2=1) then u*Real.sin a+v*Real.cos a else u*Real.cos a-v*Real.sin a


/-- Every legal schedule implements the mathematical tensor formula, with
source access safety, successful execution and the complete exterior frame. -/
theorem kernel_correct (r : Request) (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    VeriTile.Spec.Real (realIO r X Y qs ⊨ (fun xs => packOutput r Y (result r (xs 0)))) := by
  have proved := IOCorrectness.implements ideal r X Y different qs all id id
  have target_eq : IOCorrectness.target ideal r X Y qs id id =
      (fun xs => packOutput r Y (result r (xs 0))) := by
    funext xs o
    have h : o=0 := Fin.eq_zero o
    subst o
    funext j
    exact ideal_result r (logicalInput r (xs 0)) _ _ _
  rw [target_eq] at proved
  exact proved

end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.ReaderBefore
