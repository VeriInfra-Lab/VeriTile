import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Binding
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.FPEquiv
noncomputable section
open Checked Binding

/-- The finite source execution computes every returned FP32 byte. Includes
actual original/returned Tensor descriptors, optional state, exact saved context,
input bytes and all exterior bytes. Numerical assumptions are local scalars. -/
theorem rwkv_native (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (x : Inputs) (plan : Plans c) (scale : BitVec 32) (pythonScale : PythonNumber)
    (caller : VeriTile.Torch.Device) (layout : Layout c) (alloc : Allocations c x layout.base)
    (direct : DirectInputs c x layout.base) (memory : Bytes) (admitted : admit c x=true) :
    ∃ result,invoke a compiler c x plan scale pythonScale caller layout alloc memory=some result ∧
      NativeResult a a compiler c x plan scale pythonScale caller layout alloc memory result :=
  complete a compiler c x plan scale pythonScale caller layout alloc direct memory admitted

theorem rwkv_native_if_local (actual target : Native.Instructions)
    (localAgreement : Native.InstructionAgreement actual target) (compiler : Native.Compiler) (c : Shape)
    (x : Inputs) (plan : Plans c) (scale : BitVec 32) (pythonScale : PythonNumber)
    (caller : VeriTile.Torch.Device) (layout : Layout c) (alloc : Allocations c x layout.base)
    (direct : DirectInputs c x layout.base) (memory : Bytes) (admitted : admit c x=true) :
    ∃ result,invoke actual compiler c x plan scale pythonScale caller layout alloc memory=some result ∧
      NativeResult actual target compiler c x plan scale pythonScale caller layout alloc memory result :=
  complete_if_local actual target localAgreement compiler c x plan scale pythonScale caller layout alloc direct memory admitted

/-- Public helper, including its sentinel selection and forward-only traversal.
`bind` is only the scalar Python-number to FP32 argument conversion. -/
theorem rwkv_helper_native (a : Native.Instructions) (compiler : Native.Compiler) (shape : Shape)
    (x : Inputs) (plan : Plans {shape with reverse:=false})
    (requested defaultScale : PythonNumber) (bind : PythonNumber → BitVec 32)
    (caller : VeriTile.Torch.Device) (layout : Layout {shape with reverse:=false})
    (alloc : Allocations {shape with reverse:=false} x layout.base)
    (direct : DirectInputs {shape with reverse:=false} x layout.base) (memory : Bytes)
    (admitted : admit {shape with reverse:=false} x=true) :
    ∃ result,invoke a compiler {shape with reverse:=false} x plan
        (bind (selectScale requested defaultScale)) (selectScale requested defaultScale) caller layout alloc memory=some result ∧
      NativeResult a a compiler {shape with reverse:=false} x plan
        (bind (selectScale requested defaultScale)) (selectScale requested defaultScale) caller layout alloc memory result :=
  rwkv_native a compiler {shape with reverse:=false} x plan _ _ caller layout alloc direct memory admitted

#axiomsClean rwkv_helper_native

#axiomsClean rwkv_native
#axiomsClean rwkv_native_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.FPEquiv
