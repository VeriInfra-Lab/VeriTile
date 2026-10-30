import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Host
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Whole
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned Source VeriTile.Bench.Repairs.AttentionKernelAligned.Full VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def Footprint (p : Parameters) (off : Nat) : Prop :=
  ∃ b<p.batches,∃ h<p.heads,∃ n<p.length,∃ d<p.head,off=logicalAddress p b h n d

theorem outside_iff (p : Parameters) (r : RegionName) (off : Nat) :
    OutsideLogical p r off ↔ r≠p.output ∨ ¬Footprint p off := by
  simp [OutsideLogical,Footprint]

theorem access_bounds (p : Parameters) (g : Guard p) (storage : TensorStorage p)
    (x y c r d j : Nat) (hy : y<p.batches*p.heads) (hc : c<p.blocks) :
    ((x*p.blockM+r<p.length ∧ d<p.head) → offset (qStrides p) (y/p.heads) (y%p.heads) (x*p.blockM+r) d<storage.sizeQ) ∧
    ((c*p.blockN+j<p.length ∧ d<p.head) → offset (kStrides p) (y/p.heads) (y%p.heads) (c*p.blockN+j) d<storage.sizeK) ∧
    ((c*p.blockN+j<p.length ∧ d<p.head) → offset (vStrides p) (y/p.heads) (y%p.heads) (c*p.blockN+j) d<storage.sizeV) ∧
    (x*p.blockM+r<p.length → offset (bStrides p) (y/p.heads) (y%p.heads) (x*p.blockM+r) c<storage.sizeBias) ∧
    (x*p.blockM+r<p.length → offset (bStrides p) (y/p.heads) (y%p.heads) (x*p.blockM+r) (j%p.biasLast+p.biasLast)<storage.sizeBias) ∧
    ((x*p.blockM+r<p.length ∧ d<p.head) → offset (oStrides p) (y/p.heads) (y%p.heads) (x*p.blockM+r) d<storage.sizeOut) := by
  obtain ⟨hH,hb,hh⟩:=head_decode p y hy
  have hbias : c<128 := by
    have hn:=blocks_capacity p g
    omega
  have hlast : j%p.biasLast+p.biasLast<128 := by
    rw [g.selected.biasLast]
    omega
  exact ⟨fun h=>storage.query _ hb _ hh _ h.1 _ h.2,
    fun h=>storage.key _ hb _ hh _ h.1 _ h.2,
    fun h=>storage.value _ hb _ hh _ h.1 _ h.2,
    fun h=>storage.bias _ hb _ hh _ h _ hbias,
    fun h=>storage.bias _ hb _ hh _ h _ hlast,
    fun h=>storage.output _ hb _ hh _ h.1 _ h.2⟩

inductive Buffer where | query | key | value | bias | output deriving DecidableEq
def extent (storage : TensorStorage p) : Buffer → Nat
  | .query=>storage.sizeQ | .key=>storage.sizeK | .value=>storage.sizeV | .bias=>storage.sizeBias | .output=>storage.sizeOut
def width (_ : Buffer) : Nat := 2
def region (p : Parameters) : Buffer → RegionName
  | .query=>p.q | .key=>p.k | .value=>p.v | .bias=>p.bias | .output=>p.output
def formedEnvelope (p : Parameters) : Buffer → Nat
  | .query=>strideEnvelope p (qStrides p) | .key=>strideEnvelope p (kStrides p)
  | .value=>strideEnvelope p (vStrides p) | .output=>strideEnvelope p (oStrides p)
  | .bias=>biasEnvelope p

/-- Ordinary half-storage placement; the address envelope covers inactive
lanes and advanced pointers as well as every active dereference. -/
structure Placement (p : Parameters) (storage : TensorStorage p) where
  arena : Nat
  base : Buffer → Nat
  arena64 : arena≤2^64
  bounded : ∀ r,base r+2*extent storage r≤arena
  formed64 : Nonempty p → ∀ r,base r+2*formedEnvelope p r+2<2^64
  apart : Nonempty p → ∀ r,r≠.output → ∀ a<2*extent storage r,∀ b<2*storage.sizeOut,
    base r+a≠base .output+b

theorem absolute_formed (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (ne : Nonempty p) (r : Buffer) (off k : Nat) (ho : off≤formedEnvelope p r) (hk : k<2) : pl.base r+2*off+k<2^64 := by
  have hs:=pl.formed64 ne r
  omega

variable {α : Type} [Inhabited α]
def byte (w : BitVec 16) (i : Nat) : BitVec 8 := BitVec.ofNat 8 (w.toNat/256^(i%2))

def flattenBytes (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (s : State α) (i : Fin pl.arena) : BitVec 8 :=
  if pl.base .output≤ i.val ∧ i.val<pl.base .output+2*storage.sizeOut then
    byte (encode ((i.val-pl.base .output)/2) (s.mem p.output ((i.val-pl.base .output)/2))) ((i.val-pl.base .output)%2)
  else flat i

structure FlatBinding (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (s : State α) : Prop where
  output : ∀ i : Fin pl.arena,pl.base .output≤ i.val → i.val<pl.base .output+2*storage.sizeOut →
    flat i=byte (encode ((i.val-pl.base .output)/2) (s.mem p.output ((i.val-pl.base .output)/2))) ((i.val-pl.base .output)%2)

theorem flatten_active (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (s : State α)
    (off k : Nat) (ho : off<storage.sizeOut) (hk : k<2)
    (i : Fin pl.arena) (hi : i.val=pl.base .output+2*off+k) :
    flattenBytes p storage pl encode flat s i=byte (encode off (s.mem p.output off)) k := by
  have hr : i.val-pl.base .output=2*off+k := by omega
  have hd : (2*off+k)/2=off := by omega
  have hm : (2*off+k)%2=k := by omega
  have inside : pl.base .output≤ i.val ∧ i.val<pl.base .output+2*storage.sizeOut := by omega
  rw [flattenBytes,if_pos inside,hr,hd,hm]

theorem flatten_frame (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (initial final : State α)
    (binding : FlatBinding p storage pl encode flat initial)
    (frame : ∀ off,¬Footprint p off → final.mem p.output off=initial.mem p.output off)
    (i : Fin pl.arena)
    (hu : ∀ off,Footprint p off → ∀ k<2,i.val≠pl.base .output+2*off+k) :
    flattenBytes p storage pl encode flat final i=flat i := by
  unfold flattenBytes
  split_ifs with h
  · have hu' : ¬Footprint p ((i.val-pl.base .output)/2) := by
      intro ho
      apply hu _ ho ((i.val-pl.base .output)%2) (Nat.mod_lt _ (by decide))
      omega
    rw [frame _ hu']
    exact (binding.output i h.1 h.2).symm
  · rfl

#axiomsClean access_bounds
#axiomsClean absolute_formed
#axiomsClean flatten_active
#axiomsClean flatten_frame
#specNonCircular Placement avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Whole
