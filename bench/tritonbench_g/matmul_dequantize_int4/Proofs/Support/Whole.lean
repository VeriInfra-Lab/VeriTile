import bench.tritonbench_g.matmul_dequantize_int4.Proofs.RealHost
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.QuantReal
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.QuantBounds
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Public
open VeriTile.Triton FP.Structural
noncomputable section
/-- Internal proved domain, derived from the named executable public checks
in Contract. Normal Tensor/compiler/runtime semantics remain the ordinary ABI. -/
structure Admitted (c : Call) : Prop where
  valid : c.Valid
  indices : Bounds.Signed32 c
structure QuantAdmitted (c : QuantHost.Call) (m : Physical.Bytes) : Prop where
  valid : c.Valid
  indices : QuantBounds.Signed64 c
  finite : QuantFP.FiniteDomain c.shape (QuantHost.input c m) c.tree

theorem whole_real (c : Call) (h : Admitted c) (device : Nat) (i : Fin 17)
    (initial : State ℝ) (fresh : Nat → ℝ) :
    ∃ result,RealHost.call c device i initial fresh=some result ∧ RealHost.Claim c device i initial fresh result :=
  RealHost.whole_call c h.valid device i initial fresh

theorem whole_software (c : Call) (h : Admitted c) (device : Nat) (mma : Opcode → ISA.MMA16)
    (i : Fin 17) (initial : Physical.Bytes) (fresh : Nat → Physical.Byte) :
    ∃ result,Host.call c device (Host.software mma) i initial fresh=some result ∧
      Host.Claim mma c device i initial fresh result :=
  Host.whole_call c h.valid device (Host.software mma) mma (Host.software_ports mma) i initial fresh

theorem whole_fp (c : Call) (h : Admitted c) (device : Nat) (actual : Native.Instructions)
    (mma : Opcode → ISA.MMA16) (ports : Native.Obligations actual mma)
    (i : Fin 17) (initial : Physical.Bytes) (fresh : Nat → Physical.Byte) :
    ∃ result,Host.call c device actual i initial fresh=some result ∧ Host.Claim mma c device i initial fresh result :=
  Host.whole_call c h.valid device actual mma ports i initial fresh

theorem quant_software (c : QuantHost.Call) (initial : Physical.Bytes) (h : QuantAdmitted c initial) :
    QuantHost.Post c initial (QuantHost.call c QuantFP.software initial) :=
  QuantHost.whole_call c h.valid QuantFP.software ⟨fun _ _ => rfl,fun _ _ => rfl,fun _ _ => rfl,fun _ => rfl⟩ initial

theorem quant_fp (c : QuantHost.Call) (initial : Physical.Bytes) (h : QuantAdmitted c initial)
    (actual : QuantFP.Instructions) (ports : QuantFP.Obligations actual) :
    QuantHost.Post c initial (QuantHost.call c actual initial) :=
  QuantHost.whole_call c h.valid actual ports initial

#axiomsClean whole_real
#axiomsClean whole_software
#axiomsClean whole_fp
#axiomsClean quant_software
#axiomsClean quant_fp
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Public
