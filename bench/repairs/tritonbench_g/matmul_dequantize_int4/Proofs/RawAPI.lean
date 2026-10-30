import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.RawKernel
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantRaw

/-! Retained ordinary raw/physical matrix results and the two independent
PyTorch quantizer APIs. These are supporting applications, not reader specs. -/
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Correct
open VeriTile VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Complete selected matrix-product source grid on raw strided memory. -/
theorem kernel_correct_raw (layout : RawKernel.Layout) (valid : layout.Valid) (i : Fin 17)
    (initial : State ℝ) (packed : KernelReal.Packed initial) :
    ∃ final, KernelSource.run KernelReal.scalar KernelReal.dot layout.call i initial=some final ∧
      GridExecution.Post "C" (Geometry.address (Grid.geometry (params layout.call i)))
        (fun x : Grid.Index (params layout.call i) => .mk .fp16
          (Real.mathematical layout.call (RealHost.input layout.call initial) x.1.val x.2.val))
        (fun _ => True) initial final :=
  KernelReal.whole_grid layout.call (layout.valid valid) i initial packed

/-- Quantizer source arrays, complete published memory and the exterior frame. -/
theorem quantize_raw (tree : Quantizer.Tree) (shape : Quantizer.Shape)
    (valid : shape.Valid) (covered : tree.Valid shape.group)
    (x : Nat → Nat → ℝ) (Q S Z : Nat) (qs : Q≠S) (qz : Q≠Z) (sz : S≠Z)
    (heap : QuantRaw.RealHeap) :
    ∃ values final, values=Quantizer.source (QuantReal.ops tree) shape x ∧
      final=QuantRaw.realPublish shape Q S Z heap values ∧
      QuantRaw.RealPost shape x Q S Z heap final :=
  QuantRaw.real_complete tree shape valid covered x Q S Z qs qz sz heap

#axiomsClean kernel_correct_raw
#axiomsClean quantize_raw
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Correct

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.FPEquiv
noncomputable section

/-- Complete source execution with every native output byte and exterior frame. -/
theorem kernel_equiv_raw (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (layout : RawKernel.Layout)
    (valid : layout.Valid) (i : Fin 17) (initial : Physical.Bytes) :
    ∃ final,Physical.source actual layout.call i initial=some final ∧
      Physical.Post mma layout.call i initial final :=
  Physical.whole_bytes actual mma ports layout.call (layout.valid valid) i initial

theorem quantize_raw (shape : Quantizer.Shape) (view : View) (tree : Quantizer.Tree)
    (valid : shape.Valid) (viewShape : view.rows=shape.K ∧ view.cols=shape.N)
    (viewValid : view.Valid) (covered : tree.Valid shape.group)
    (actual : QuantFP.Instructions) (ports : QuantFP.Obligations actual)
    (initial : Physical.Bytes) :
    ∃ final, final=QuantRaw.nativeRun shape view tree actual initial ∧
      QuantRaw.NativePost shape view tree initial final :=
  QuantRaw.native_complete shape view tree valid viewShape viewValid covered actual ports initial

#axiomsClean kernel_equiv_raw
#axiomsClean quantize_raw
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.FPEquiv
