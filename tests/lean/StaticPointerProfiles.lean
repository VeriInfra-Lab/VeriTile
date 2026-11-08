import VeriTile.Triton.DSL.StaticPointerProfiles
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StaticPointerProfiles
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

inductive Storage where
  | fp32 | fp16 | bf16

def Storage.dtype : Storage → SourceDType
  | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16

def signature (storage : Storage) : CallSignature where
  pointers := [("X",storage.dtype),("Y",storage.dtype)]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (storage : Storage) (X Y : RegionName) : ComputeKernel :=
  with_triton_call (signature storage) call_args [] in triton {
    value = tl.load(X)
    tl.store(Y, value.to(Y.dtype.element_ty))
  }

/-- Caller metadata preserves both actual input and output port lists. -/
theorem ports (storage : Storage) (X Y : RegionName) :
    (copied storage X Y).inputs=[X] ∧ (copied storage X Y).outputs=[Y] := by
  cases storage <;> exact ⟨rfl,rfl⟩

/-- The fp16 branch retains a typed fp16 load before any erasure. -/
theorem fp16_channel (X Y : RegionName) :
    (match (copied .fp16 X Y).surfaceBody.head? with
     | some (.assign dtype _ _ _) => dtype == .fp16
     | _=>Bool.false)=Bool.true := rfl

theorem bf16_channel (X Y : RegionName) :
    (match (copied .bf16 X Y).surfaceBody.head? with
     | some (.assign dtype _ _ _) => dtype == .bf16
     | _=>Bool.false)=Bool.true := rfl

theorem fp32_channel (X Y : RegionName) :
    (match (copied .fp32 X Y).surfaceBody.head? with
     | some (.assign dtype _ _ _) => dtype == .real
     | _=>Bool.false)=Bool.true := rfl

#axiomsClean ports
#axiomsClean fp16_channel
#axiomsClean bf16_channel
#axiomsClean fp32_channel
end
end VeriTile.Tests.StaticPointerProfiles
