import bench.repairs.tritonbench_g.llama_ff_triton.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Contract
noncomputable section

/-! The scalar-cast repair changes no wrapper check or admitted payload.
Aliases deliberately retain the original complete contract, including empty M,
unit dimensions/strides, both storage branches and nondivisible K. -/
abbrev TensorInfo := LlamaFF.Contract.TensorInfo
abbrev Argument := LlamaFF.Contract.Argument
abbrev InputContract := LlamaFF.Contract.InputContract
abbrev Admitted := LlamaFF.Contract.Admitted
abbrev check := LlamaFF.Contract.check
abbrev shape := LlamaFF.Contract.shape

theorem admitted_iff (x w v r : TensorInfo) :
    Admitted x w v r ↔ InputContract x w v r :=
  LlamaFF.Contract.python_checks_iff x w v r

#axiomsClean admitted_iff
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Contract
