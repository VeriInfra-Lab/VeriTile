import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IONative
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Misc.KernelNativeResults

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
open VeriTile Triton FP.Structural FlashAttn
noncomputable section

/-- The selected source precision and scalar ABI, including raw integer one,
all FP32 words, four reduction replicas and the original owner observation. -/
def nativeAttentionIO (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI) :
    ProgramIO Word KernelNative.Expr := PublicIO.nativeIO I x scale

/-- Independent source-to-native observation for the full O/L tensor. Arbitrary
output-stride collisions keep the actual last writer. Instructions are existing
primitive model parameters, not new FP transformation assumptions. -/
theorem whole_native (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI) :
    (nativeAttentionIO I x scale).Implements (Logical.nativeFormula I x scale) :=
  PublicIO.native_implements I x scale

#axiomsClean whole_native
#stmtSurfaceSubset whole_native ⊆
  [ProgramIO.Implements,nativeAttentionIO,Logical.nativeFormula,Layout,Checked.ScalarABI,
   KernelNative.Instructions,KernelNative.Expr,KernelNative.instInhabitedExpr,Word]
#specNonCircular Logical.nativeFormula avoiding
  [FP.Structural.exec,FP.Structural.run,ActualSource.runGrid,ActualSource.referenceGrid,ActualSource.accumulated]
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
