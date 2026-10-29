import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPSoftware
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape

/-! Public finite-precision correctness surface for four previously accepted
HGRN stages. These are execution-to-value contracts, not candidate-equivalence
claims. A concrete unfused fp32 ScalarOps instance is included; native CUDA
conformance and the full recurrent/reverse loops remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPCorrect
open VeriTile.Triton FP.Structural FPStages
noncomputable section
specification forward_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (forward BHPrev X G O t T D BD) s = some final ∧
      Post s final O t T D BD (forwardValues M s BHPrev X G t T D BD) := by
  exact FPStages.forward_fp32_stage M BHPrev X G O t T D BD s

specification dx_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dx DHPrev DO DX t T D BD) s = some final ∧
      Post s final DX t T D BD (dxValues M s DHPrev DO t T D BD) := by
  exact FPStages.dx_fp32_stage M DHPrev DO DX t T D BD s

specification dx_two_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      Post s final DX t T D BD (dxTwoValues M s DHPrev DO G t T D BD) := by
  exact FPStages.dx_two_fp32_stage M DHPrev DO G DX t T D BD s

specification dg_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      Post s final DG t T D BD (dgValues M s DHPrev DO O H0 U t T D BD) := by
  exact FPStages.dg_fp32_stage M DHPrev DO O H0 DG U t T D BD s

specification forward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (forward BHPrev X G O t T D BD) s = some final ∧
      Post s final O t T D BD (roundedForward cfg s BHPrev X G t T D BD) := by
  exact FPStages.forward_software_fp32 rest cfg BHPrev X G O t T D BD s

specification dx_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dx DHPrev DO DX t T D BD) s = some final ∧
      Post s final DX t T D BD (roundedDx cfg s DHPrev DO t T D BD) := by
  exact FPStages.dx_software_fp32 rest cfg DHPrev DO DX t T D BD s

specification dx_two_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      Post s final DX t T D BD (roundedDxTwo cfg s DHPrev DO G t T D BD) := by
  exact FPStages.dx_two_software_fp32 rest cfg DHPrev DO G DX t T D BD s

specification dg_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      Post s final DG t T D BD (roundedDg cfg s DHPrev DO O H0 U t T D BD) := by
  exact FPStages.dg_software_fp32 rest cfg DHPrev DO O H0 DG U t T D BD s

specification forward_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (forward "carry" "input" "gate" "out" 0 1 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  exact FPStages.forward_nonempty_rounding rest

#axiomsClean forward_fp32_stage
#axiomsClean dx_fp32_stage
#axiomsClean dx_two_fp32_stage
#axiomsClean dg_fp32_stage
#axiomsClean forward_software_fp32
#axiomsClean dx_software_fp32
#axiomsClean dx_two_software_fp32
#axiomsClean dg_software_fp32
#axiomsClean forward_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPCorrect
