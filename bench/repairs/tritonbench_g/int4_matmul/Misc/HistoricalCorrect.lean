import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectMatmulReal
import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectQuantReal
import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectUnpack
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Real
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperReal
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperBounds
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperHost
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Source
namespace VeriTile.Bench.Repairs.Int4Matmul
open Helpers Host
noncomputable section

structure MatmulRealClaim (x : Arguments) (t : Tile) (ws : List Work) (initial : State ℝ)
    (fresh : Tensor) (caller : Nat) (result : Host.Return ℝ) : Prop where
  identity : result.tensor=outputTensor x fresh
  layout : result.tensor.Canonical .fp32 x.shape.m x.shape.n x.input.device
  device : result.callerDevice=caller
  source : result.state=Source.prepared x.shape t realOps initial ws (zeroOutput x.shape realOps initial)
  values : ∀ r c,r<x.shape.m → c<x.shape.n →
    result.state.values .output (cAddress x.shape r c)=mathematical x.shape initial r c
  frame : ∀ r a,r≠.output ∨ x.shape.m*x.shape.n≤a → result.state.values r a=initial.values r a
  packedFrame : result.state.packed=initial.packed
  indices : IndexDomain x.shape t
  actualStrides : StrideBinding x t

 theorem matmul_real (x : Arguments) (valid : Contract x) (t : Tile) (member : t∈sourceTiles)
    (ws : List Work) (complete : Complete x.shape t ws) (initial : State ℝ)
    (fresh : Tensor) (layout : FreshOutput x fresh) (caller : Nat) :
    MatmulRealClaim x t ws initial fresh caller (hostCall x t realOps ws fresh caller initial) := by
  have ht := all_tiles_valid t member
  refine ⟨rfl,return_geometry x valid fresh layout.layout,rfl,?_,?_,?_,?_,guarded_indices x.shape valid.indices t member,contract_strides x valid t member⟩
  · exact (Source.prepared_execute x.shape t realOps initial (zeroOutput x.shape realOps initial) ws
      (fun w hw => (complete_active x.shape t valid.shape ht ws complete w).mp hw)
      (zero_readonly x.shape realOps initial)).symm
  · exact whole_real x.shape t valid.shape ht initial ws complete
  · exact call_frame x.shape t valid.shape ht realOps ws complete initial
  · exact (execute_readonly x.shape t realOps ws initial _ (zero_readonly x.shape realOps initial)).2

structure QuantRealClaim (s : Shape) (order : ReductionOrder s.group) (initial : State ℝ)
    (result : Quantized ℝ) : Prop where
  shape : result.weightShape=(s.k/8,s.n) ∧ result.scaleShape=(s.k/s.group,s.n) ∧
    result.zeroShape=(s.k/s.group,s.n/8) ∧ result.groupSize=s.group
  weights : ∀ p n,p<s.k/8 → n<s.n → result.weights (p*s.n+n)=weightSpec s HelperReal.arithmetic order initial p n
  zeros : ∀ g p,g<s.k/s.group → p<s.n/8 → result.zeros (g*(s.n/8)+p)=zeroSpec s HelperReal.arithmetic order initial g p
  scales : ∀ g n,g<s.k/s.group → n<s.n → ∀ hi lo,
    HelperReal.GroupRange (fun i : Fin s.group => input s initial g n i) hi lo →
    result.scales (g*s.n+n)=(hi-lo)/15
  zeroFormula : ∀ g n hi lo,HelperReal.GroupRange (fun i : Fin s.group => input s initial g n i) hi lo →
    zeroNibble s HelperReal.arithmetic order initial g n=HelperReal.mathZero hi lo
  weightFormula : ∀ k n hi lo,HelperReal.GroupRange (fun i : Fin s.group => input s initial (k/s.group) n i) hi lo →
    weightNibble s HelperReal.arithmetic order initial k n=HelperReal.mathWeight (initial.values .weight (k*s.n+n)) hi lo

theorem quantize_real (s : Shape) (hs : s.Valid) (order : ReductionOrder s.group) (initial : State ℝ) :
    QuantRealClaim s order initial (quantize s HelperReal.arithmetic order initial) := by
  refine ⟨⟨rfl,rfl,rfl,rfl⟩,quantize_result s hs _ order initial,?_,?_,?_,?_⟩
  · intro g p _ hp;exact quantize_zeros s hs _ order initial g p hp
  · intro g n _ hn hi lo range
    rw [quantize_scales s hs _ order initial g n hn]
    exact HelperReal.scale_independent s order initial g n hi lo range
  · exact HelperReal.zero_independent s order initial
  · exact HelperReal.weight_independent s order initial

structure UnpackRealClaim (s : Shape) (initial : State ℝ) : Prop where
  values : ∀ k n,k<s.k → n<s.n → unpackLoop s HelperReal.halfReal initial k n (s.k/s.group)=
    ((logicalQ s initial k n-logicalZ s initial k n:Int):ℝ)*logicalScale s initial k n
  address : ∀ k n,k<s.k → n<s.n → n*s.k+k<s.n*s.k

theorem unpack_real (s : Shape) (hs : s.Valid) (initial : State ℝ) : UnpackRealClaim s initial := by
  refine ⟨?_,unpack_address s⟩
  intro k n hk _
  exact HelperReal.unpack_independent s hs initial k n hk

/-- Public real theorem includes all three source APIs. The matrix component is
ordinary GEMM; quantization uses independent group ranges and nearest-even clamp. -/
structure WholeReal : Prop where
  matmul : ∀ x,(valid : Contract x) → ∀ t,t∈sourceTiles → ∀ ws,Complete x.shape t ws →
    ∀ initial fresh,FreshOutput x fresh → ∀ caller,
      MatmulRealClaim x t ws initial fresh caller (hostCall x t realOps ws fresh caller initial)
  quantizer : ∀ s,s.Valid → ∀ order initial,QuantRealClaim s order initial (quantize s HelperReal.arithmetic order initial)
  unpacker : ∀ s,s.Valid → ∀ initial,UnpackRealClaim s initial

theorem whole_real_apis : WholeReal := ⟨matmul_real,quantize_real,unpack_real⟩
#axiomsClean matmul_real
#axiomsClean quantize_real
#axiomsClean unpack_real
#axiomsClean whole_real_apis
end
end VeriTile.Bench.Repairs.Int4Matmul

namespace VeriTile.Bench.Repairs.Int4Matmul.Specification
abbrev matmul_dequantize_int4_s2_wrapper_correct := @Direct.matmul_real_complete
abbrev quantize_int4_wrapper_correct := @Direct.quant_real_complete
abbrev unpack_int4_wrapper_correct := @Direct.unpack_real_complete
#axiomsClean matmul_dequantize_int4_s2_wrapper_correct
#axiomsClean quantize_int4_wrapper_correct
#axiomsClean unpack_int4_wrapper_correct
end VeriTile.Bench.Repairs.Int4Matmul.Specification
