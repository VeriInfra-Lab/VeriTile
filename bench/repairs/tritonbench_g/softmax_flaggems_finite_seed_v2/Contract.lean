import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.Contract
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev TensorContract := SoftmaxFlaggemsFiniteSeed.TensorContract
abbrev ForwardContract := SoftmaxFlaggemsFiniteSeed.ForwardContract
abbrev GradientContract := SoftmaxFlaggemsFiniteSeed.GradientContract
abbrev checkForward := SoftmaxFlaggemsFiniteSeed.checkForward
abbrev checkGradient := SoftmaxFlaggemsFiniteSeed.checkGradient
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
