import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.TorchBinding
namespace VeriTile.Bench.Repairs.MatmulDequantize.FPEquiv
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Triton FP.Structural Host Checked ByteBridge NativeWhole
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem gptq (a : Arguments) (cfg : FirstConfig) (t : TensorBinding.Tensors a .gptq)
    (h : InputContract a .gptq) (plan : Native.Plan) (actual target : Native.WordISA)
    (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .gptq t (FirstGrid.geometry (first a cfg)) 2 Bool.true plan actual m (gptqAction a cfg h m)=.ok (some result) ∧
      Guarantee a .gptq cfg t (FirstGrid.geometry (first a cfg)) 2 plan target m (gptqReference a cfg m) result :=
  NativeWhole.gptq a cfg t h plan actual target ports m

theorem s2 (a : Arguments) (t : TensorBinding.Tensors a .s2) (h : InputContract a .s2)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .s2 t (SecondGrid.geometry (second a)) 2 Bool.true plan actual m (s2Action a h m)=.ok (some result) ∧
      Guarantee a .s2 .m64n256 t (SecondGrid.geometry (second a)) 2 plan target m (s2Reference a m) result :=
  NativeWhole.s2 a t h plan actual target ports m

theorem dequant32 (a : Arguments) (t : TensorBinding.Tensors a .dequant32) (h : InputContract a .dequant32)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .dequant32 t (DequantGrid32.geometry (Host.dequant32 a)) 4 Bool.false plan actual m (dequant32Action a h m)=.ok (some result) ∧
      Guarantee a .dequant32 .m64n256 t (DequantGrid32.geometry (Host.dequant32 a)) 4 plan target m (dequant32Reference a m) result :=
  NativeWhole.dequant32 a t h plan actual target ports m

theorem dequant16 (a : Arguments) (t : TensorBinding.Tensors a .dequant16) (h : InputContract a .dequant16)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .dequant16 t (DequantGrid16.geometry (Host.dequant16 a)) 2 Bool.true plan actual m (dequant16Action a h m)=.ok (some result) ∧
      Guarantee a .dequant16 .m64n256 t (DequantGrid16.geometry (Host.dequant16 a)) 2 plan target m (dequant16Reference a m) result :=
  NativeWhole.dequant16 a t h plan actual target ports m

theorem s1 (a : Arguments) (t : TensorBinding.Tensors a .s1) (h : InputContract a .s1)
    (api : TorchBinding.NativeMM a t) (plan : Native.Plan) (actual target : Native.WordISA)
    (ports : Numerics actual target) (before : Memory) :
    ∃ result,TorchBinding.invoke a t h api plan actual before=.ok (some result) ∧ TorchBinding.Guarantee a t api plan target before result :=
  TorchBinding.native a t h api plan actual target ports before

#axiomsClean gptq
#axiomsClean s2
#axiomsClean dequant32
#axiomsClean dequant16
#axiomsClean s1
end
end VeriTile.Bench.Repairs.MatmulDequantize.FPEquiv
