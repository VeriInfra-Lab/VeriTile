import bench.tritonbench_g.softmax_triton1.Contract

namespace VeriTile.Bench.Repairs.TritonSoftmax
abbrev InputContract := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.InputContract
abbrev InputError := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.InputError
abbrev checkInput := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.checkInput
abbrev checkTensorInput := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.checkTensorInput
abbrev AdmittedArgument := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.AdmittedArgument
end VeriTile.Bench.Repairs.TritonSoftmax
