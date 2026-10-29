import bench.repairs.tritonbench_g.swiglu_triton.Kernels

/-! Complete executable identity with the established typed FP32 source family.
The pasted helper and all four writer quotations retain their full signatures,
casts, memory events, floating order and candidate index products. -/
namespace VeriTile.Bench.Repairs.SwigluTriton.BodyCopy
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluTriton
set_option maxRecDepth 20000

theorem forward_complete_program (a b c : RegionName) (stride cols block : Nat) :
    forwardKernel a b c stride cols block =
      typedFwd .fp32 .fp32 Bool.false a b c stride cols block := rfl

#axiomsClean forward_complete_program

theorem forward_candidate_complete_program (a b c : RegionName) (stride cols block : Nat) :
    forwardIndexCommuted a b c stride cols block =
      typedFwd .fp32 .fp32 Bool.true a b c stride cols block := rfl

#axiomsClean forward_candidate_complete_program

theorem backward_complete_program (dc a b : RegionName) (stride cols block : Nat) :
    backwardKernel dc a b stride cols block =
      typedBwd .fp32 .fp32 .fp32 Bool.false dc a b stride cols block := rfl

#axiomsClean backward_complete_program

theorem backward_candidate_complete_program (dc a b : RegionName) (stride cols block : Nat) :
    backwardIndexCommuted dc a b stride cols block =
      typedBwd .fp32 .fp32 .fp32 Bool.true dc a b stride cols block := rfl

#axiomsClean backward_candidate_complete_program

end VeriTile.Bench.Repairs.SwigluTriton.BodyCopy
