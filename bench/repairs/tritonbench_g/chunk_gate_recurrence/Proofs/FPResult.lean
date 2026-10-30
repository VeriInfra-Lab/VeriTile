import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Results

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Public
open Checked Binding Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory (Key)
namespace Bytes
abbrev Byte := BitVec 8
abbrev encode := VeriTile.Bench.Repairs.RotaryEmb.Physical.encode

/-- Ordinary local little-endian four-byte packing, restricted to aligned
FP32 words. It contains no kernel, numerical operation, or output formula. -/
structure Codec {M : Type} (api : API M Word Byte) : Prop where
  read : ∀ m (key : Key),key.address%4=0 → ∀ i : Fin 4,
    api.observe m key.owner (key.address+i.val)=encode (api.read m key) i
end Bytes
noncomputable section

theorem key_aligned (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (x : Cell)
    (active : Machine.Active c seed x.region x.program x.time x.lane) :
    (key c seed runtime x).address%4=0 := by
  have positive : 0<extent c runtime.seedStrides x.region := by
    have bound := active_offset c seed runtime.abi x.region x.program x.time x.lane active
    change 0 ≤ Machine.offset c runtime.abi x.region x.program x.time x.lane ∧
      Machine.offset c runtime.abi x.region x.program x.time x.lane < (extent c runtime.seedStrides x.region : Int) at bound
    omega
  have aligned := (backing_checked c seed runtime accepted x.region
    (active_present c seed _ _ _ _ active) positive).2.1
  simp [key,Nat.add_mod,aligned]

theorem native_backward_source {M β : Type} (api : API M Word β) (isa : ISA)
    (laws : LocalConformance isa allOperands) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (trees : Trees c) (m : M) :
    nativeBackward api isa c seed runtime trees m=backward api (wordOps isa) c seed runtime trees m := by
  unfold nativeBackward backward
  rw [SplitNative.backward_complete isa laws,←Source.backward_complete,SplitNative.scalar_ext isa laws]

structure FPClaim {M : Type} (api : API M Word Bytes.Byte) (isa : ISA) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (trees : Trees c) (call : Invocation api (wordOps isa) c seed runtime)
    (out : Forward M) (grad : Backward M) : Prop extends Claim api (wordOps isa) c seed runtime trees call out grad where
  scalarProfile : isa=SourceFP.scalar
  outputBytes : ∀ p t j,p<c.programs → t<c.N → j<1024 → ∀ i : Fin 4,
    api.observe out.memory (key c seed runtime ⟨.output,p,t,j⟩).owner
      ((key c seed runtime ⟨.output,p,t,j⟩).address+i.val)=
        Bytes.encode (Reference.forward (wordOps isa) c call.forwardInputs seed p t j) i
  gradientBytes : ∀ x,x∈stageCells c seed Bool.false → ∀ i : Fin 4,
    api.observe grad.memory (key c seed runtime x).owner ((key c seed runtime x).address+i.val)=
      Bytes.encode (Source.backwardTarget (wordOps SourceFP.scalar) c trees seed
        (mirror api c seed runtime call.backwardPrepared) x.region x.program x.time x.lane) i

theorem fp_complete {M : Type} (api : API M Word Bytes.Byte) (codec : Bytes.Codec api)
    (isa : ISA) (laws : LocalConformance isa allOperands) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c)
    (call : Invocation api (wordOps isa) c seed runtime) :
    FPClaim api isa c seed runtime trees call (forward api (wordOps isa) c seed runtime call.forwardPrepared)
      (nativeBackward api isa c seed runtime trees call.backwardPrepared) := by
  have source : Claim api (wordOps isa) c seed runtime trees call
      (forward api (wordOps isa) c seed runtime call.forwardPrepared)
      (nativeBackward api isa c seed runtime trees call.backwardPrepared) := by
    rw [native_backward_source api isa laws]
    exact complete api (wordOps isa) c seed runtime accepted trees call
  refine ⟨source,SplitNative.scalar_ext isa laws,?_,?_⟩
  · intro p t j hp ht hj i
    rw [codec.read _ _ (key_aligned c seed runtime accepted ⟨.output,p,t,j⟩ ⟨hp,ht,hj⟩),
      source.output p t j hp ht hj]
  · intro x hx i
    have write := (stage_mem c seed Bool.false x).mp hx
    have active := (active_written c seed _ _ _ _ write.1).2
    rw [codec.read _ _ (key_aligned c seed runtime accepted x active),
      native_read api isa laws c seed runtime accepted trees call.backwardPrepared x write.1 write.2]

#axiomsClean fp_complete
#specNonCircular Bytes.Codec avoiding [Source.run,Source.backwardRun,Reference.forward]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Public
