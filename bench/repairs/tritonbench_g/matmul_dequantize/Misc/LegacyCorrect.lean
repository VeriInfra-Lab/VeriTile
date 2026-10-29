import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.TorchBinding
namespace VeriTile.Bench.Repairs.MatmulDequantize.Correct
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Triton FP.Structural Host Checked ByteBridge RealWhole
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem gptq (a : Arguments) (cfg : FirstConfig) (t : TensorBinding.Tensors a .gptq) (h : InputContract a .gptq)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .gptq t (gptqAction a cfg h raw heap)=.ok (some result) ∧
      Guarantee a .gptq cfg t (FirstGrid.geometry (first a cfg)) Bool.true (entry a raw heap) (gptqReference a cfg raw heap) result :=
  RealWhole.gptq a cfg t h raw heap

theorem s2 (a : Arguments) (t : TensorBinding.Tensors a .s2) (h : InputContract a .s2)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .s2 t (s2Action a h raw heap)=.ok (some result) ∧
      Guarantee a .s2 .m64n256 t (SecondGrid.geometry (second a)) Bool.true (entry a raw heap) (s2Reference a raw heap) result :=
  RealWhole.s2 a t h raw heap

theorem dequant32 (a : Arguments) (t : TensorBinding.Tensors a .dequant32) (h : InputContract a .dequant32)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .dequant32 t (dequant32Action a h raw heap)=.ok (some result) ∧
      Guarantee a .dequant32 .m64n256 t (DequantGrid32.geometry (Host.dequant32 a)) Bool.false (entry a raw heap) (dequant32Reference a raw heap) result :=
  RealWhole.dequant32 a t h raw heap

theorem dequant16 (a : Arguments) (t : TensorBinding.Tensors a .dequant16) (h : InputContract a .dequant16)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .dequant16 t (dequant16Action a h raw heap)=.ok (some result) ∧
      Guarantee a .dequant16 .m64n256 t (DequantGrid16.geometry (Host.dequant16 a)) Bool.true (entry a raw heap) (dequant16Reference a raw heap) result :=
  RealWhole.dequant16 a t h raw heap

theorem s1 (api : TorchMM.RealAPI) (a : Arguments) (t : TensorBinding.Tensors a .s1)
    (h : InputContract a .s1) (raw : Memory) (heap : RealWhole.Heap) :
    ∃ result,RealWhole.invoke a .s1 t
      (TorchMM.s1RealRun api a (admitted a .s1 h)
        (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (RealWhole.entry a raw heap))=.ok (some result) ∧
      RealWhole.Guarantee a .s1 .m64n256 t (SecondGrid.geometry (second a)) Bool.true
        (RealWhole.entry a raw heap)
        (fun x => Real.rawMatmul (RealFirst.inputs (first a .m64n256) (RealWhole.entry a raw heap)) a.K a.group x.1.val x.2.val) result :=
  TorchBinding.real api a t h raw heap

#axiomsClean gptq
#axiomsClean s2
#axiomsClean dequant32
#axiomsClean dequant16
#axiomsClean s1
end
end VeriTile.Bench.Repairs.MatmulDequantize.Correct
