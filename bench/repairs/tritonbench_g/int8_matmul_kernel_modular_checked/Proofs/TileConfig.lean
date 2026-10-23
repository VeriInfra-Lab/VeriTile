import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileSource

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section

/-- Source pointer origins are the admitted views' data pointers; strides are
in elements. Storage-offset bounds remain obligations of the wrapper bridge. -/
structure Layout where
  (input weight output : RegionName)
  (inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride : Nat)

def parameters (layout : Layout) (s : Shape) (c : Fin 15) : Parameters where
  input := layout.input
  weight := layout.weight
  output := layout.output
  rows := s.m
  columns := s.n
  inner := s.k
  inputRowStride := layout.inputRowStride
  inputInnerStride := layout.inputInnerStride
  weightInnerStride := layout.weightInnerStride
  weightColumnStride := layout.weightColumnStride
  outputRowStride := layout.outputRowStride
  outputColumnStride := layout.outputColumnStride
  blockRows := (configs c).bm
  blockColumns := (configs c).bn
  blockInner := (configs c).bk
  groupRows := (configs c).gm

def StaticAdmission (p : Parameters) : Prop := p.inner % (4*p.blockInner) = 0

/-- All fifteen retained configurations use precisely their own source BK
divisibility assertion. No stronger common K guard is introduced. -/
theorem all_config_assertions (layout : Layout) (s : Shape) (c : Fin 15)
    (legal : s.legal c) : StaticAdmission (parameters layout s c) := legal

theorem source_steps (layout : Layout) (s : Shape) (c : Fin 15) :
    (parameters layout s c).steps = s.steps c := rfl

#axiomsClean all_config_assertions
#axiomsClean source_steps
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
