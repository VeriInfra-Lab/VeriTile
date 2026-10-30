import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 5000000

def cellByte (v : Stored (BitVec 32) (BitVec 16)) (i : Nat) : Byte :=
  match v with
  | .inl b => if i=0 then b else 0
  | .inr (.inl f) => BitVec.ofNat 8 (f.toNat/2^(8*i))
  | .inr (.inr h) => BitVec.ofNat 8 (h.toNat/2^(8*i))
structure RawByteABI (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) where
  bytes : σ → Address → Byte
  width : Nat → Nat
  positive : ∀ storage,0<width storage
  decode : ∀ state storage offset,bytes state (storage,offset)=
    cellByte (abi.observe state (storage,offset/width storage)) (offset%width storage)

theorem native_full_byte_frame (actual : StorageOps (BitVec 32) (BitVec 16))
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (storage offset : Nat) (outside : ∀ x : Cell s,(storage,offset/raw.width storage)≠outputAddress b x) :
    raw.bytes ((hostCode s binding).native abi state).2 (storage,offset)=raw.bytes state (storage,offset) := by
  rw [raw.decode,raw.decode,hostCode_eq,binding.lowered,native_grid]
  rw [run_frame actual p b selectedPlans selectedReductions binding.schedule.events (abi.observe state)
    (storage,offset/raw.width storage) outside]
theorem native_output_bytes (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (x : Cell s) (i : Fin (raw.width (outputAddress b x).1)) :
    raw.bytes ((hostCode s binding).native abi state).2
      ((outputAddress b x).1,raw.width (outputAddress b x).1*(outputAddress b x).2+i.val)=
      cellByte (wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) i.val := by
  have h := (bound_native_complete actual reference leaves abi p b domain.fresh domain.nonoverlap binding state).1 x
  rw [raw.decode]
  simp only [Nat.mul_add_div (raw.positive _),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
    Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt,hostCode_eq]
  change cellByte (abi.observe (boundExecute actual abi binding state).2 (outputAddress b x)) i.val=_
  rw [h]

#axiomsClean native_full_byte_frame
#axiomsClean native_output_bytes
end
end VeriTile.Bench.Repairs.LlamaFF
