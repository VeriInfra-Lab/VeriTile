import bench.tritonbench_g.fast_ce_loss.Proofs.KernelBoundary

/-! Prepared kernel argument Tensors, actual byte storage, raw Python scalar
payloads, and launch geometry. Historical host contracts are preserved in
Proofs/HostContract.lean. No host execution is claimed by this entrypoint. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.Contract

def checkedSourceSHA256 : String := "c848b2bb538b8b0800a069e564b2a796dcdf23f0b2688e0fc8b6f73a5ecf07ff"
def bodySourceSHA256 : String := "bd8421753644480fb676f8b76e351be1db156b791376f8b0d5a925febeefc382"

export KernelSource (Parameters)
export KernelContract.Storage (Word Byte Allocation TensorStorage CoherentBytes wordAt intAt)
export KernelContract (FloatArgument LabelArgument MatrixView VectorView FlatView ParametersValid
  Launch LaunchState ForwardViews BackwardViews ForwardInputs BackwardInputs tensorValue tensorLabel)
export KernelABI (PythonScalar Options Conversions LiteralProfile)
export KernelLaunch (ForwardGeometry ChunkedGeometry BackwardGeometry forward_covered flat_vector flat_matrix column_mask)
export KernelBoundary (ForwardAddresses BackwardAddresses forward_addresses backward_addresses forward_logit_leaf forward_label_bytes)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FastCeLoss.Contract
