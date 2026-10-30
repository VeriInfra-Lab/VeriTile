import bench.tritonbench_g.layer_norm_welfold.Contract
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.LayerNormWelfold.InputBinding
open VeriTile.Triton FP.Structural

/-- Ordinary Tensor API interpretation of the observed strides, shape, offset
and data pointer. These laws concern original input storage, never a result.
In particular, no extra `is_contiguous` guard is added to the Python domain. -/
structure Layout (t : VeriTile.Torch.FP32.Tensor) (m : Admission.Metadata) : Prop where
  metadata : t.metadata = m.tensor
  offset : t.storageOffset = m.storageOffset
  vector : ∀ n stride, m.tensor.shape = [n] → m.strides = [stride] →
    ∀ i, i < n → t.physicalIndex i = t.storageOffset + i * stride
  matrix : ∀ n d sr sc, m.tensor.shape = [n,d] → m.strides = [sr,sc] →
    ∀ row, row < n → ∀ col, col < d →
      t.physicalIndex (col + d * row) = t.storageOffset + row * sr + col * sc

/-- The actual launch pointer plus its FP32 byte displacement. -/
def address (t : VeriTile.Torch.FP32.Tensor) (i : Nat) : Nat := t.dataPtr.toNat + 4 * i

/-- Storage bounds exclude modular pointer wrap on every actual load. -/
theorem address_logical (read : Nat → VeriTile.Torch.FP32.Word) (t : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws t)
    (notNeg : t.metadata.isNeg = Bool.false) (i : Nat) (hi : i < t.metadata.numel)
    (linear : t.physicalIndex i = t.storageOffset + i) :
    address t i = t.storage.address (t.physicalIndex i) ∧
    address t i < 2^64 ∧ read (address t i) = VeriTile.Torch.FP32.logicalValue read t laws i := by
  have hb := laws.inBounds i hi
  have span := t.storage.spanFits
  have ha : t.storage.address (t.physicalIndex i) < 2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  have base : t.storage.address t.storageOffset < 2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address] at ha ⊢
    omega
  have ptr := congrArg BitVec.toNat (laws.pointer (by omega))
  simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt base] at ptr
  have he : address t i = t.storage.address (t.physicalIndex i) := by
    simp only [address, ptr, VeriTile.Torch.FP32.Storage.address, linear]
    omega
  exact ⟨he, he ▸ ha, by simp [he, VeriTile.Torch.FP32.logicalValue, notNeg]⟩

/-- The native 64-bit pointer addition denotes the same active byte address. -/
theorem native_address (t : VeriTile.Torch.FP32.Tensor) (i : Nat)
    (bound : address t i < 2^64) :
    (t.dataPtr + BitVec.ofNat 64 (4 * i)).toNat = address t i := by
  rw [BitVec.toNat_add, BitVec.toNat_ofNat]
  have offset : 4 * i < 2^64 := by unfold address at bound; omega
  rw [Nat.mod_eq_of_lt offset]
  exact Nat.mod_eq_of_lt bound

/-- One shared physical reader preserves equal and overlapping read-only input
storage. The regional snapshot merely interprets these original pointer loads. -/
def state {α : Type} (read : Nat → VeriTile.Torch.FP32.Word) (decode : VeriTile.Torch.FP32.Word → α)
    (x w bias : VeriTile.Torch.FP32.Tensor) (scratch : State α) : State α :=
  { scratch with mem := fun q i =>
      if q = "X" then .mk .real (decode (read (address x i)))
      else if q = "W" then .mk .real (decode (read (address w i)))
      else if q = "Bias" then .mk .real (decode (read (address bias i)))
      else scratch.mem q i }

/-- Original logical Tensor values, interpreted only at valid logical indices. -/
def logical {α : Type} (read : Nat → VeriTile.Torch.FP32.Word) (decode : VeriTile.Torch.FP32.Word → α)
    (t : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws t) : Nat → α :=
  fun i => decode (VeriTile.Torch.FP32.logicalValue read t laws i)

structure Observations {α : Type} [Inhabited α] (read : Nat → VeriTile.Torch.FP32.Word) (decode : VeriTile.Torch.FP32.Word → α)
    (x w bias : VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x) (lw : VeriTile.Torch.FP32.TensorLaws w) (lb : VeriTile.Torch.FP32.TensorLaws bias)
    (s : State α) (S D : Nat) : Prop where
  xValues : ∀ row, row < S → ∀ col, col < D →
    (s.mem "X" (col + D * row)).read .real = logical read decode x lx (col + D * row)
  wValues : ∀ col, col < D → (s.mem "W" col).read .real = logical read decode w lw col
  biasValues : ∀ col, col < D → (s.mem "Bias" col).read .real = logical read decode bias lb col
  xPointer : ∀ row, row < S → ∀ col, col < D →
    address x (col + D * row) = x.storage.address (x.physicalIndex (col + D * row)) ∧
    address x (col + D * row) < 2^64 ∧
    (x.dataPtr + BitVec.ofNat 64 (4 * (col + D * row))).toNat = address x (col + D * row)
  wPointer : ∀ col, col < D → address w col = w.storage.address (w.physicalIndex col) ∧ address w col < 2^64 ∧
    (w.dataPtr + BitVec.ofNat 64 (4 * col)).toNat = address w col
  biasPointer : ∀ col, col < D → address bias col = bias.storage.address (bias.physicalIndex col) ∧ address bias col < 2^64 ∧
    (bias.dataPtr + BitVec.ofNat 64 (4 * col)).toNat = address bias col

theorem observed {α : Type} [Inhabited α] (read : Nat → VeriTile.Torch.FP32.Word) (decode : VeriTile.Torch.FP32.Word → α)
    (x w bias : VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x) (lw : VeriTile.Torch.FP32.TensorLaws w) (lb : VeriTile.Torch.FP32.TensorLaws bias)
    (mx mw mb : Admission.Metadata) (bx : Layout x mx) (bw : Layout w mw) (bb : Layout bias mb)
    (S D : Nat) (h : Admission.Contract mw mb mx S D) (scratch : State α) :
    Observations read decode x w bias lx lw lb (state read decode x w bias scratch) S D := by
  have hx : ∀ row, row < S → ∀ col, col < D →
      address x (col + D * row) = x.storage.address (x.physicalIndex (col + D * row)) ∧
      address x (col + D * row) < 2^64 ∧
      read (address x (col + D * row)) = VeriTile.Torch.FP32.logicalValue read x lx (col + D * row) := by
    intro row hr col hc
    apply address_logical read x lx
    · simpa [bx.metadata] using h.noNeg.2.2
    · simp only [bx.metadata, VeriTile.Torch.TensorMetadata.numel, h.xShape,
        List.prod_cons, List.prod_nil, Nat.mul_one]
      nlinarith
    · simpa [Nat.mul_comm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        bx.matrix S D D 1 h.xShape h.strides.1 row hr col hc
  have hw : ∀ col, col < D → address w col = w.storage.address (w.physicalIndex col) ∧
      address w col < 2^64 ∧ read (address w col) = VeriTile.Torch.FP32.logicalValue read w lw col := by
    intro col hc
    apply address_logical read w lw
    · simpa [bw.metadata] using h.noNeg.1
    · simpa [bw.metadata, VeriTile.Torch.TensorMetadata.numel, h.affineShape.1] using hc
    · simpa using bw.vector D 1 h.affineShape.1 h.strides.2.1 col hc
  have hb : ∀ col, col < D → address bias col = bias.storage.address (bias.physicalIndex col) ∧
      address bias col < 2^64 ∧ read (address bias col) = VeriTile.Torch.FP32.logicalValue read bias lb col := by
    intro col hc
    apply address_logical read bias lb
    · simpa [bb.metadata] using h.noNeg.2.1
    · simpa [bb.metadata, VeriTile.Torch.TensorMetadata.numel, h.affineShape.2] using hc
    · simpa using bb.vector D 1 h.affineShape.2 h.strides.2.2 col hc
  exact ⟨fun r hr c hc => congrArg decode (hx r hr c hc).2.2,
    fun c hc => congrArg decode (hw c hc).2.2,
    fun c hc => congrArg decode (hb c hc).2.2,
    fun r hr c hc => ⟨(hx r hr c hc).1, (hx r hr c hc).2.1,
      native_address x _ (hx r hr c hc).2.1⟩,
    fun c hc => ⟨(hw c hc).1, (hw c hc).2.1, native_address w _ (hw c hc).2.1⟩,
    fun c hc => ⟨(hb c hc).1, (hb c hc).2.1, native_address bias _ (hb c hc).2.1⟩⟩

#axiomsClean address_logical
#axiomsClean native_address
#axiomsClean observed
end VeriTile.Bench.Repairs.LayerNormWelfold.InputBinding
