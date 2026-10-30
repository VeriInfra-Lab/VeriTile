import bench.tritonbench_g.attention_forward_triton.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def Footprint (p : Parameters) (off : Nat) : Prop :=
  ∃ b<p.batches,∃ h<p.heads,∃ n<p.length,∃ d<96,off=logicalAddress p b h n d

theorem active_footprint {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (pid : Program) (hp : pid∈programs (grid p)) (i : TileIndex [128,128])
    (hi : active p (start (grid p) pid s) i) :
    Footprint p (address p (start (grid p) pid s) i) := by
  obtain ⟨hx,hy,hz⟩:=(mem_programs (grid p) pid.1 pid.2.1 pid.2.2).mp hp
  obtain ⟨hH,hb,hh⟩:=head_decode p pid.2.1 hy
  refine ⟨pid.2.1/p.heads,hb,pid.2.1%p.heads,hh,pid.1*128+i.1.val,hi.1,i.2.1.val,hi.2,?_⟩
  rfl

/-- Bounds for every dereferenced element of the six memory operations in the
inlined source. Invalid tensor lanes are guarded and use literal zero. Scale
loads are unmasked, so their bounds follow from the actual grid and loop index. -/
theorem access_bounds (p : Parameters) (storage : TensorStorage p)
    (x y row r d j : Nat) (hx : x<(p.length+127)/128) (hy : y<p.batches*p.heads)
    (hr : row<p.length) :
    ((x*128+r<p.length ∧ d<96) →
      y/p.heads*p.q.batch+y%p.heads*p.q.head+(x*128+r)*p.q.row+d*p.q.dim<storage.sizeQ) ∧
    ((j<p.length-row ∧ d<96) →
      y/p.heads*p.k.batch+y%p.heads*p.k.head+d*p.k.dim+(row+j)*p.k.row<storage.sizeK) ∧
    ((j<p.length-row ∧ d<96) →
      y/p.heads*p.v.batch+y%p.heads*p.v.head+(row+j)*p.v.row+d*p.v.dim<storage.sizeV) ∧
    y*((p.length+127)/128)+x<storage.sizeQS ∧
    y*((p.length+63)/64)+row/64<storage.sizeKS ∧
    ((x*128+r<p.length ∧ d<96) →
      y/p.heads*p.o.batch+y%p.heads*p.o.head+(x*128+r)*p.o.row+d*p.o.dim<storage.sizeOut) := by
  obtain ⟨hH,hb,hh⟩:=head_decode p y hy
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro h;exact storage.query _ hb _ hh _ h.1 _ h.2
  · intro h
    have he:=storage.key _ hb _ hh (row+j) (by omega) d h.2
    simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using he
  · intro h;exact storage.value _ hb _ hh (row+j) (by omega) d h.2
  · rw [storage.scales.1]
    have hm:=Nat.mul_le_mul_right ((p.length+127)/128) (show y+1≤p.batches*p.heads by omega)
    rw [Nat.add_mul,Nat.one_mul] at hm
    omega
  · rw [storage.scales.2]
    have hrow : row/64<(p.length+63)/64 := by omega
    have hm:=Nat.mul_le_mul_right ((p.length+63)/64) (show y+1≤p.batches*p.heads by omega)
    rw [Nat.add_mul,Nat.one_mul] at hm
    omega
  · intro h;exact storage.output _ hb _ hh _ h.1 _ h.2

variable {α : Type} [Inhabited α]
theorem host_result_full (M : Program → Algebra α) (dot : Program → Dot α)
    (p : Parameters) (storage : TensorStorage p) (abi : InputABI p) (before : State α)
    (alloc : Allocation p abi before) (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke M dot p abi before alloc work=some result ∧ result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk .fp16
          (answer (M (n/128,b*p.heads+h,0)) (dot (n/128,b*p.heads+h,0)) p
            (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → result.state.mem r off=alloc.state.mem r off) := by
  have uniq : Unique p (grid p) work alloc.state := by
    intro a ha b hb i j hi hj he
    exact grid_unique p storage alloc.state a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
  obtain ⟨final,hr,hv,hf⟩:=run_correct M dot p (separated p storage) (grid p) work alloc.state uniq
  refine ⟨⟨final,alloc.output⟩,by simp [invoke,hr],rfl,?_,?_⟩
  · intro b hb h hh n hn d hd
    have ha : active p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd) :=
      ⟨by change n/128*128+n%128<p.length;omega,hd⟩
    have value:=hv _ (legal.mem_iff.mp (grid_covers p b h n d hb hh hn hd).1) _ ha
    rw [physical_address p alloc.state b h n d hh hd] at value
    exact value
  · intro r off hu
    apply hf r off
    rcases hu with ho|hu
    · exact Or.inl ho
    · right
      intro pid hp i hi he
      exact hu (he ▸ active_footprint p alloc.state pid (legal.mem_iff.mpr hp) i hi)

inductive Buffer where | query | key | value | queryScale | keyScale | output
  deriving DecidableEq
def extent (storage : TensorStorage p) : Buffer → Nat
  | .query=>storage.sizeQ | .key=>storage.sizeK | .value=>storage.sizeV
  | .queryScale=>storage.sizeQS | .keyScale=>storage.sizeKS | .output=>storage.sizeOut
def width : Buffer → Nat | .output=>2 | _=>4
def region (p : Parameters) : Buffer → RegionName
  | .query=>p.Q | .key=>p.K | .value=>p.V | .queryScale=>p.QS | .keyScale=>p.KS | .output=>p.Out

/-- Ordinary byte placement/allocation contract. Bases are absolute addresses;
the 64-bit limit also covers formed masked addresses, not just dereferences. -/
structure Placement (p : Parameters) (storage : TensorStorage p) where
  arena : Nat
  base : Buffer → Nat
  arena64 : arena≤2^64
  bounded : ∀ r,base r+width r*extent storage r≤arena
  formed64 : ∀ r,base r+4*envelope p<2^64
  apart : ∀ r,r≠.output → ∀ a<width r*extent storage r,∀ b<2*storage.sizeOut,
    base r+a≠base .output+b

theorem absolute_formed (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (r : Buffer) (off : Nat) (ho : off<envelope p) : pl.base r+width r*off<2^64 := by
  have hw : width r≤4 := by cases r <;> decide
  have hm:=Nat.mul_le_mul hw (Nat.le_of_lt ho)
  exact lt_of_le_of_lt (Nat.add_le_add_left hm _) (pl.formed64 r)

def byte (w : BitVec 16) (i : Nat) : BitVec 8 := BitVec.ofNat 8 (w.toNat/256^(i%2))

def flattenBytes (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (s : State α) (i : Fin pl.arena) : BitVec 8 :=
  if pl.base .output≤ i.val ∧ i.val<pl.base .output+2*storage.sizeOut then
    byte (encode ((i.val-pl.base .output)/2) (s.mem p.Out ((i.val-pl.base .output)/2))) ((i.val-pl.base .output)%2)
  else flat i

structure FlatBinding (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (s : State α) : Prop where
  output : ∀ i : Fin pl.arena,pl.base .output≤ i.val → i.val<pl.base .output+2*storage.sizeOut →
    flat i=byte (encode ((i.val-pl.base .output)/2) (s.mem p.Out ((i.val-pl.base .output)/2))) ((i.val-pl.base .output)%2)

theorem flatten_active (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (s : State α)
    (off k : Nat) (ho : off<storage.sizeOut) (hk : k<2)
    (i : Fin pl.arena) (hi : i.val=pl.base .output+2*off+k) :
    flattenBytes p storage pl encode flat s i=byte (encode off (s.mem p.Out off)) k := by
  have hr : i.val-pl.base .output=2*off+k := by omega
  have hd : (2*off+k)/2=off := by omega
  have hm : (2*off+k)%2=k := by omega
  have inside : pl.base .output≤ i.val ∧ i.val<pl.base .output+2*storage.sizeOut := by omega
  rw [flattenBytes,if_pos inside,hr,hd,hm]

theorem flatten_frame (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (encode : Nat → Cell α → BitVec 16) (flat : Fin pl.arena → BitVec 8) (initial final : State α)
    (binding : FlatBinding p storage pl encode flat initial)
    (frame : ∀ off,¬Footprint p off → final.mem p.Out off=initial.mem p.Out off)
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
#axiomsClean host_result_full
#axiomsClean absolute_formed
#axiomsClean flatten_active
#axiomsClean flatten_frame
#specNonCircular Placement avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Whole
