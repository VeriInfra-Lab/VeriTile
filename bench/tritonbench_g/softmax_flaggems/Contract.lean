import bench.tritonbench_g.softmax_flaggems.Proofs.SoftmaxFlaggemsFiniteSeed_Contract
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev TensorContract := SoftmaxFlaggemsFiniteSeed.TensorContract
abbrev ForwardContract := SoftmaxFlaggemsFiniteSeed.ForwardContract
abbrev GradientContract := SoftmaxFlaggemsFiniteSeed.GradientContract
abbrev checkForward := SoftmaxFlaggemsFiniteSeed.checkForward
abbrev checkGradient := SoftmaxFlaggemsFiniteSeed.checkGradient
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
