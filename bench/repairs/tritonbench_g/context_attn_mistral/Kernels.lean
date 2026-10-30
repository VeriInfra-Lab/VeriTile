import bench.tritonbench_g.context_attn_mistral.Proofs.Program
import bench.tritonbench_g.context_attn_mistral.Proofs.HostInterface

/-! Direct checked FP32 candidate. The supplied output is mutated in place;
no floating payload is resolved, cloned or copied. Integer widths are widened
to i64 and the equivalent causal/window difference predicate avoids signed
subtraction overflow. Erasing only widths and normalizing that predicate
leaves this full typed source. All numerical instructions and guards remain. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
abbrev repairedKernel := Program.kernel

def addressBits : Nat := 64
def blockSize : Nat := 128
end VeriTile.Bench.Repairs.ContextAttnMistral
