import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Polynomial

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Defect
noncomputable section

/-- Original g_0 contribution to g_x, from CONST009/CONST040 and VAR06/08/24/26.
All other upstream channels are zero in the exact counterexample. -/
def originalChannelZeroX (x z : ℝ) : ℝ :=
  (11.6340690431164 : ℝ)*x^4+(11.6340690431164 : ℝ)*z^4+
    (-69.8044142586986 : ℝ)*x^2*z^2

theorem exact_counterexample :
    originalChannelZeroX 1 0-Polynomial.jacobian 0 0 1 0 0=(-1/20000000000000 : ℝ) := by
  norm_num [originalChannelZeroX,Polynomial.jacobian]

theorem original_backward_not_derivative :
    originalChannelZeroX 1 0≠Polynomial.jacobian 0 0 1 0 0 := by
  have h:=exact_counterexample
  intro equal
  rw [equal,sub_self] at h
  norm_num at h

#axiomsClean exact_counterexample
#axiomsClean original_backward_not_derivative
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Defect
