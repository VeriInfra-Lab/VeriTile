import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Correct
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
#check kernel_correct
#check kernel_native
#check TensorBridge.kernel_correct
#check TensorBridge.kernel_native
#check TensorBridge.empty_call
example {α : Type} (M : VeriTile.Triton.FP.Structural.Algebra α)
    (R : TokenAttnMistral.SumPrimitive α) : (TokenAttnMistral.withSum M R).cast=M.cast := rfl
example (word : Word) : TokenAttnMistral.Native.algebra.fp32Load (.bits word)=.load (.bits word) := rfl
#axiomsClean python_checks_iff
#axiomsClean launch_bounds
#axiomsClean kernel_correct
#axiomsClean kernel_native
#axiomsClean TensorBridge.kernel_correct
#axiomsClean TensorBridge.kernel_native
#axiomsClean TensorBridge.empty_call
#specNonCircular realTarget avoiding [kernel,VeriTile.Triton.FP.Structural.exec,TokenAttnMistral.Grid.run,TokenAttnMistral.outputValues]
#specNonCircular logicalRealTarget avoiding [kernel,VeriTile.Triton.FP.Structural.exec,TokenAttnMistral.Grid.run,TokenAttnMistral.outputValues]
#specNonCircular NativeTarget.reference avoiding [kernel,VeriTile.Triton.FP.Structural.exec,TokenAttnMistral.Grid.run,TokenAttnMistral.outputValues]
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
