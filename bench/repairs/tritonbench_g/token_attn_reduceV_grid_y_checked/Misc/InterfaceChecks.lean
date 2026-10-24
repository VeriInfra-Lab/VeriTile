import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Correct
import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
#axiomsClean grid_real
#axiomsClean grid_native
#check TensorBridge.kernel_correct
#check TensorBridge.kernel_native
#check TensorBridge.empty_call
example {α : Type} (M : VeriTile.Triton.FP.Structural.Algebra α)
    (R : TokenAttnReduceV.SumPrimitive α) : (TokenAttnReduceV.withSum M R).cast=M.cast := rfl
example (word : Word) : TokenAttnReduceV.Native.algebra.fp32Load (.bits word)=.load (.bits word) := rfl
#axiomsClean python_checks_iff
#axiomsClean launch_bounds
#axiomsClean TensorBridge.kernel_correct
#axiomsClean TensorBridge.kernel_native
#axiomsClean TensorBridge.empty_call
#specNonCircular realTarget avoiding [kernel,VeriTile.Triton.FP.Structural.exec,TokenAttnReduceV.Grid.run,TokenAttnReduceV.outputValues]
#specNonCircular logicalRealTarget avoiding [kernel,VeriTile.Triton.FP.Structural.exec,TokenAttnReduceV.Grid.run,TokenAttnReduceV.outputValues]
#specNonCircular NativeTarget.reference avoiding [kernel,VeriTile.Triton.FP.Structural.exec,TokenAttnReduceV.Grid.run,TokenAttnReduceV.outputValues]
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
