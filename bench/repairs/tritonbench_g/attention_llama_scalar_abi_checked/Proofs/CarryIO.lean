import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.CarryIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage NativeSchedule
noncomputable section
open scoped Classical

@[simp] theorem schedule_length (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) :
    (schedule C arg B c).length=Fintype.card (Program B c) := List.length_ofFn

theorem schedule_get (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (k : Fin (schedule C arg B c).length) :
    (schedule C arg B c)[k]=invocation C arg B c
      (program B c ⟨k.val,by simpa only [schedule_length] using k.isLt⟩) := by
  exact List.get_ofFn (fun i=>invocation C arg B c (program B c i)) k

theorem invocation_initial (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (hD : 0<c.D) (p : Program B c) (s : State (Term α)) :
    (invocation C arg B c p).initial s=
      bindScale (Tensor.initialFor c s (Backing.anchor hD p)) (Native.argument C arg) :=
  GridGeometry.invocation_initial C arg B c hD p s

#axiomsClean invocation_initial
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryIO
