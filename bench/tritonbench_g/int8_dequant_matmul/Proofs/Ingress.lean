import bench.tritonbench_g.int8_dequant_matmul.Proofs.Native

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section

/-- Same-storage raw/resolved tensor admission. No finite/positive/nonzero
restriction is imposed on scales or bias, and no int32 no-overflow gate is
imposed on the int8 dot. Same-dtype aliases among read-only inputs are allowed. -/
structure TypedInputs (s : Shape) (b : Buffers) (hasBias : Bool) (m : Memory Word32 Word16) : Prop where
  aTyped : ∀ r k,r<s.M → k<s.K → ∃ value,m (b.a,b.aOffset+r*s.K+k)=.inl value
  bTyped : ∀ k n,k<s.K → n<s.N → ∃ value,m (b.b,b.bOffset+k*s.N+n)=.inl value
  xTyped : ∀ r,r<s.M → ∃ value,m (b.x,b.xOffset+r)=.inr (.inl value)
  wTyped : ∀ n,n<s.N → ∃ value,m (b.w,b.wOffset+n)=.inr (.inl value)
  biasTyped : hasBias=Bool.true → ∀ n,n<s.N → ∃ value,m (b.bias,b.biasOffset+n)=.inr (.inl value)

theorem raw_signed8 (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (state : σ) (storage offset : Nat) (kind : raw.dtype storage=.int8)
    (typed : ∃ value,abi.observe state (storage,offset)=.inl value) :
    read8 (abi.observe state) (storage,offset)=raw.bytes state (storage,offset) := by
  obtain ⟨value,typed⟩ := typed
  rw [raw.decode,kind]
  simp [DType.bytes,cellByte,read8,typed]

theorem raw_float32 (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (state : σ) (storage offset : Nat) (i : Fin 4) (kind : raw.dtype storage=.fp32)
    (typed : ∃ value,abi.observe state (storage,offset)=.inr (.inl value)) :
    (read32 a (abi.observe state) (storage,offset)).bits.extractLsb' (8*i.val) 8=
      raw.bytes state (storage,4*offset+i.val) := by
  obtain ⟨value,typed⟩ := typed
  rw [raw.decode,kind]
  simp only [DType.bytes,Nat.mul_add_div (by decide : 0<4),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
    Nat.mul_add_mod,Nat.mod_eq_of_lt i.isLt,typed,read32,cellByte]

/-- The domain contract identifies the actual resolved buffers used by the
source wrapper, and relates every raw float/byte input to the same native
storage observed by the proof. Output freshness is allocation-level. -/
structure PublicInput (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (s : Shape) (b : Buffers) (hasBias : Bool) (state : σ) : Prop where
  typed : TypedInputs s b hasBias (abi.observe state)
  aKind : raw.dtype b.a=.int8
  bKind : raw.dtype b.b=.int8
  xKind : raw.dtype b.x=.fp32
  wKind : raw.dtype b.w=.fp32
  biasKind : hasBias=Bool.true → raw.dtype b.bias=.fp32
  outKind : raw.dtype b.out=.fp16
  fresh : b.Fresh

theorem input_A_from_same_bytes
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (input : PublicInput abi raw s b hasBias state) (r k : Nat) (hr : r<s.M) (hk : k<s.K) :
    read8 (abi.observe state) (b.a,b.aOffset+r*s.K+k)=raw.bytes state (b.a,b.aOffset+r*s.K+k) :=
  raw_signed8 abi raw state _ _ input.aKind (input.typed.aTyped r k hr hk)
theorem input_B_from_same_bytes
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (input : PublicInput abi raw s b hasBias state) (k n : Nat) (hk : k<s.K) (hn : n<s.N) :
    read8 (abi.observe state) (b.b,b.bOffset+k*s.N+n)=raw.bytes state (b.b,b.bOffset+k*s.N+n) :=
  raw_signed8 abi raw state _ _ input.bKind (input.typed.bTyped k n hk hn)

#axiomsClean raw_signed8
#axiomsClean raw_float32
#axiomsClean input_A_from_same_bytes
#axiomsClean input_B_from_same_bytes
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
