import bench.repairs.tritonbench_g.diag_ssm_triton.Correct
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.DiagSsmTriton.TensorBridge
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

abbrev input_binding := Admission.Binding
abbrev read_view_state := @Admission.bindReadViews
alias read_view_logical := Admission.bindReadViews_logical
alias input_layouts := Admission.layouts
alias input_resolved := Admission.resolved
alias input_aligned := Admission.aligned
alias native_index_bounds := Admission.native_shape
alias python_checks_iff_preconditions := Contract.checkArguments_iff
alias python_gradient_check_iff := Contract.checkGradient_iff

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.TensorBridge
