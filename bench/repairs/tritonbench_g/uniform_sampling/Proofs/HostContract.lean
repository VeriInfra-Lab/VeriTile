import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Contract
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Protocol
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Source

namespace VeriTile.Bench.Repairs.UniformSampling
open VeriTile.Torch VeriTile.Triton
noncomputable section

/-- Same supplied Tensor and its active contiguous addresses. `View` contains
ordinary backing extents; no random output value is part of this binding. -/
structure TensorBinding (output : TensorMetadata) (v : View n) : Prop where
  numel : n=output.numel
  device : v.device=output.device.index
  negative : v.negative=output.isNeg
  pointer : ∀ i,i<n→v.address i=output.dataPtr.toNat+4*i

def selectedGenerator (defaults : Nat→Nat) (device : Nat) : GeneratorArgument → Nat
  | .absent => defaults device
  | .supplied data => data.object
  | .invalid => 0

def guarded (output : TensorMetadata) (generator : GeneratorArgument)
    (lowerIsFloat upperIsFloat : Bool) (body : α) : Except InputError α := do
  checkInput output generator lowerIsFloat upperIsFloat
  return body

theorem guarded_eq (contract : InputContract output generator lowerIsFloat upperIsFloat) (body : α) :
    guarded output generator lowerIsFloat upperIsFloat body=.ok body := by
  simp [guarded,(checkInput_iff output generator lowerIsFloat upperIsFloat).mpr contract]

theorem checked_counter_count (contract : InputContract output generator lowerIsFloat upperIsFloat)
    (binding : TensorBinding output (v : View n)) : counterCount n<2^64 := by
  have h:=Bounds.padded_bound n
  have b:=ReservationProof.block_properties n
  have size:=contract.extent
  rw [←binding.numel] at size
  have blockBound : block n≤1024 := by
    unfold block VeriTile.Bench.TritonBenchG.UniformSampling.SourceDomain.wrapperBlock
    split <;> omega
  omega

/-- Source bounds include inactive clamped pointer lanes. Alignment comes
from the ordinary fresh Torch FP32 allocation; no random result is assumed. -/
theorem checked_native_addresses (scratch : Scratch n) (aligned : scratch.base%4=0)
    (hs:block n∣start) (x : IndexProof.Location (block n)) (hx:x∈IndexProof.schedule n start) :
    let j:=4*x.1.done+min (outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2) (chunkSize n x.1-1)
    j<n ∧ scratchAddress scratch j+4≤scratch.extent ∧ scratchAddress scratch j+4≤2^64 ∧
    scratchAddress scratch j%4=0 ∧ x.2.1<2^31 := by
  obtain ⟨a,b,c⟩:=Bounds.pointer hs hx scratch
  have ix:=Bounds.native_indices hs hx
  have hb:=ReservationProof.block_properties n
  have pid : x.2.1<2^31 := by
    have blockPositive : 1<block n := by rcases hb.1 with h|h <;> omega
    nlinarith [Nat.zero_le x.2.2.2.val]
  exact ⟨a,b,c,by simp [scratchAddress,Nat.add_mod,aligned],pid⟩

open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory

theorem checked_empty (contract : InputContract output generator lowerIsFloat upperIsFloat)
    (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View 0) (_binding : TensorBinding output v) (scratch : Scratch 0)
    (which : Nat) (lib : Host.Runtime) (stops : Host.Stops) (s : Host.State Store) :
    guarded output generator lowerIsFloat upperIsFloat
      ((Host.entry api M scalars v scratch which lib stops).run s)=.ok ⟨s,.ok v.handle,[]⟩ := by
  rw [guarded_eq contract,Host.empty]

theorem checked_all_outcomes (contract : InputContract output generator lowerIsFloat upperIsFloat)
    (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (_binding : TensorBinding output v) (scratch : Scratch n)
    (which : Nat) (lib : Host.Runtime) (stops : Host.Stops) (s : Host.State Store) :
    guarded output generator lowerIsFloat upperIsFloat
      ((Host.entry api M scalars v scratch which lib stops).run s)=
        .ok ((Host.entry api M scalars v scratch which lib stops).run s) ∧
    (∀ r b,r≠scratch.region→Execution.Outside v r b→
      api.byte (((Host.entry api M scalars v scratch which lib stops).run s).state.memory r) b=
        api.byte (s.memory r) b) ∧
    (∀ other,other≠which→
      ((Host.entry api M scalars v scratch which lib stops).run s).state.generators other=s.generators other) :=
  ⟨guarded_eq contract _,Host.all_outcomes_frame api M scalars v scratch which lib stops s,
    fun other h=>Protocol.all_outcomes_other_generator api M scalars v scratch which other lib stops s h⟩

#axiomsClean checkInput_iff
#axiomsClean checked_counter_count
#axiomsClean checked_native_addresses
end
end VeriTile.Bench.Repairs.UniformSampling
