import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Floating
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardValues
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardKernel

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

abbrev Byte := BitVec 8
abbrev Word := BitVec 32

def byteOf (word : Word) (j : Fin 4) : Byte := BitVec.ofNat 8 (word.toNat / 256^j.val)

/-- Current native allocation metadata. Allocation identities in `World.live`
are ghost history; only `active` allocations own native addresses now. The
native runtime can retire an allocation and reuse its address in a later state.
Every byte, including a partial final FP32 word, has its own exact location. -/
structure Layout (σ : Type) where
  active : σ → RegionName → Prop
  buffer : σ → RegionName → Native.Buffer
  locate : σ → Nat → Nat → Option (RegionName × Nat × Fin 4)
  other : σ → Nat → Nat → Byte
  owner : ∀ s r i j, active s r → 4*i+j.val<(buffer s r).capacity →
    locate s (buffer s r).device ((buffer s r).base+4*i+j.val)=some (r,i,j)
  located : ∀ s d a r i j, locate s d a=some (r,i,j) →
    active s r ∧ d=(buffer s r).device ∧ a=(buffer s r).base+4*i+j.val ∧
      4*i+j.val<(buffer s r).capacity
  addressSpace : ∀ s r, active s r → (buffer s r).base+(buffer s r).capacity≤2^64
  aligned : ∀ s r, active s r → (buffer s r).base%4=0

/-- Physical bytes use the raw FP32-bit observation of each four-byte chunk.
Integer/bool observations are separate typed channels; their active reads are
coupled below. No unused typed projection is asserted to survive a store. -/
def bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (s : World α σ) (device address : Nat) : Byte :=
  match layout.locate s.native device address with
  | none => layout.other s.native device address
  | some (r,i,j) => byteOf (encode ((s.memory r i).read .real)) j

def rawWord {α σ : Type} [Inhabited α] (encode : α → Word) (s : World α σ)
    (r : RegionName) (i : Nat) : Word := encode ((s.memory r i).read .real)

/-- The two FP32 word observations covering one actual signed int64 cell.
This supports simultaneous int64 and FP32 aliases of a saved label allocation. -/
def rawInt64 {α σ : Type} [Inhabited α] (encode : α → Word) (s : World α σ)
    (r : RegionName) (i : Nat) : Int :=
  ((rawWord encode s r (2*i+1)).append (rawWord encode s r (2*i))).toInt

def IntRead {α σ : Type} [Inhabited α] (encode : α → Word) (s : World α σ)
    (r : RegionName) (i : Nat) : Prop := (s.memory r i).read .int=rawInt64 encode s r i

def BoolRead {α σ : Type} [Inhabited α] (encode : α → Word) (s : World α σ)
    (r : RegionName) (i : Nat) : Prop :=
  (s.memory r i).read .bool=decide (byteOf (rawWord encode s r (i/4)) ⟨i%4,by omega⟩≠0)

def Covers {σ : Type} (layout : Layout σ) (native : σ) (r : RegionName) (i device address : Nat) : Prop :=
  device=(layout.buffer native r).device ∧
    (layout.buffer native r).base+4*i≤address ∧ address<(layout.buffer native r).base+4*i+4

def TensorCovers {σ : Type} (layout : Layout σ) (native : σ) (t : Tensor) (device address : Nat) : Prop :=
  ∃ i, i<t.numel ∧ Covers layout native t.region (t.index i) device address

theorem raw_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (s : World α σ) (r : RegionName) (i : Nat) (j : Fin 4)
    (active : layout.active s.native r) (backed : 4*i+j.val<(layout.buffer s.native r).capacity) :
    bytes layout encode s (layout.buffer s.native r).device ((layout.buffer s.native r).base+4*i+j.val)=
      byteOf (rawWord encode s r i) j := by
  simp only [bytes,layout.owner s.native r i j active backed,rawWord]

theorem byte_local {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (a b : World α σ) (r : RegionName) (activeA : layout.active a.native r)
    (activeB : layout.active b.native r) (sameBuffer : layout.buffer a.native r=layout.buffer b.native r)
    (same : ∀ i, a.memory r i=b.memory r i) (offset : Nat) (inside : offset<(layout.buffer a.native r).capacity) :
    bytes layout encode a (layout.buffer a.native r).device ((layout.buffer a.native r).base+offset)=
      bytes layout encode b (layout.buffer a.native r).device ((layout.buffer a.native r).base+offset) := by
  let j : Fin 4 := ⟨offset%4,by omega⟩
  have coordinate : 4*(offset/4)+j.val=offset := by dsimp [j]; omega
  have ha := raw_bytes layout encode a r (offset/4) j activeA (by simpa only [coordinate] using inside)
  have hb := raw_bytes layout encode b r (offset/4) j activeB (by simpa only [← sameBuffer,coordinate] using inside)
  rw [← sameBuffer] at hb
  have equalWord : rawWord encode a r (offset/4)=rawWord encode b r (offset/4) := by simp only [rawWord,same]
  rw [equalWord] at ha
  simpa only [Nat.add_assoc,coordinate] using ha.trans hb.symm

theorem write_byte_frame {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (s : World α σ) (r : RegionName) (i : Nat) (v : α) (device address : Nat)
    (outside : ¬Covers layout s.native r i device address) :
    bytes layout encode {s with memory:=Source.write s.memory r i (.mk .real v)} device address=
      bytes layout encode s device address := by
  unfold bytes
  cases hl : layout.locate s.native device address with
  | none => rfl
  | some loc =>
    rcases loc with ⟨u,k,j⟩
    have h := layout.located s.native device address u k j hl
    have different : ¬(u=r ∧ k=i) := by
      rintro ⟨rfl,rfl⟩
      exact outside ⟨h.2.1,by omega,by have bound := j.isLt; omega⟩
    simp only [Source.write,if_neg different]

theorem int_read_frame {α σ : Type} [Inhabited α] (encode : α → Word)
    (s : World α σ) (written labels : RegionName) (i j : Nat) (v : α)
    (separate : labels≠written) (coupled : IntRead encode s labels i) :
    IntRead encode {s with memory:=Source.write s.memory written j (.mk .real v)} labels i := by
  simpa only [IntRead,rawInt64,rawWord,Source.write,if_neg (not_and.mpr (fun same => False.elim (separate same)))] using coupled

theorem forward_byte_frame {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (M : FP32 α) (p : Parameters) (s : World α σ) (count device address : Nat)
    (separation : Grid.ForwardSeparation p)
    (outside : ∀ q∈Grid.tasks p.rows count, ∀ r i, Grid.forwardWrites p q (r,i) →
      ¬Covers layout s.native r i device address) :
    bytes layout encode {s with memory:=Grid.forward M p s.memory count} device address=
      bytes layout encode s device address := by
  unfold bytes
  cases hl : layout.locate s.native device address with
  | none => rfl
  | some loc =>
    rcases loc with ⟨r,i,j⟩
    have h := layout.located s.native device address r i j hl
    have frame := Grid.forward_complete_frame M p s.memory count separation r i (by
      intro q hq hw
      exact outside q hq r i hw ⟨h.2.1,by omega,by have bound := j.isLt; omega⟩)
    dsimp only
    rw [frame]

theorem backward_byte_frame {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (M : FP32 α) (p : Parameters) (s : World α σ) (count device address : Nat)
    (separation : Grid.BackwardSeparation p)
    (outside : ∀ q∈Grid.tasks p.rows count, ∀ r i, Grid.backwardWrites p q (r,i) →
      ¬Covers layout s.native r i device address) :
    bytes layout encode {s with memory:=Grid.backward M p s.memory count} device address=
      bytes layout encode s device address := by
  unfold bytes
  cases hl : layout.locate s.native device address with
  | none => rfl
  | some loc =>
    rcases loc with ⟨r,i,j⟩
    have h := layout.located s.native device address r i j hl
    have frame := Grid.backward_complete_frame M p s.memory count separation r i (by
      intro q hq hw
      exact outside q hq r i hw ⟨h.2.1,by omega,by have bound := j.isLt; omega⟩)
    dsimp only
    rw [frame]

/-- A retained caller allocation keeps its address while all of its backing
bytes are framed, even if other, dead allocations have been reused. -/
theorem preserved_allocation_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (obs : Observations α σ) (a b : World α σ) (r : RegionName)
    (frame : Preserves obs a b) (live : a.live r) (activeA : layout.active a.native r)
    (activeB : layout.active b.native r) (sameBuffer : layout.buffer a.native r=layout.buffer b.native r) :
    ∀ offset, offset<(layout.buffer a.native r).capacity →
      bytes layout encode b (layout.buffer a.native r).device ((layout.buffer a.native r).base+offset)=
        bytes layout encode a (layout.buffer a.native r).device ((layout.buffer a.native r).base+offset) := by
  intro offset inside
  exact (byte_local layout encode a b r activeA activeB sameBuffer
    (fun i => (frame.memory r i live).symm) offset inside).symm

#axiomsClean raw_bytes
#axiomsClean byte_local
#axiomsClean write_byte_frame
#axiomsClean int_read_frame
#axiomsClean forward_byte_frame
#axiomsClean backward_byte_frame
#axiomsClean preserved_allocation_bytes
#specNonCircular Layout avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
