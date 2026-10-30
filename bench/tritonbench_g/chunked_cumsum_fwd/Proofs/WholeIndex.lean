import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeFlat

/-! A finite signed32 specialization and complete FP32 byte intervals. The
range domain lists source intermediates, including masked head/time lanes.
It is not a GPU compiler theorem or a change to the original source. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
open VeriTile.Triton Math Program Initialization
noncomputable section
set_option maxHeartbeats 1200000

theorem chunk_start_lt (c : Host) (l : Launch) (hl : c.valid l) :
    l.chunk*c.chunk < c.sequence := by
  have hh := hl.2.1
  change l.chunk < (c.sequence+c.chunk-1)/c.chunk at hh
  have h := (Nat.lt_div_iff_mul_lt c.chunkPositive).1 hh
  have hp := c.chunkPositive
  omega

/-- Source integer constants, IDs, products, pointer-offset sums and masks.
The bounds quantify over all physical tile lanes, including inactive lanes. -/
def integerTerms (c : Host) (l : Launch) (i j : Nat) : List Nat :=
  let p := c.layout
  let h := head p l i
  let dtb := l.batch*p.dtBatch
  let dtc := l.chunk*p.chunk*p.dtTime
  let oub := l.batch*p.outBatch
  let ouc := l.chunk*p.outChunk
  [c.batches,c.sequence,c.heads,c.chunk,c.BH,c.BC,c.chunks,c.headBlocks,
   p.dtBatch,p.dtTime,p.dtHead,p.rateHead,p.biasHead,
   p.outBatch,p.outChunk,p.outHead,p.outColumn,
   l.batch,l.chunk,l.head,i,j,l.chunk*p.chunk,c.sequence-l.chunk*p.chunk,
   min c.chunk (c.sequence-l.chunk*c.chunk),l.head*p.BH,h,
   dtb,dtc,dtb+dtc,h*p.dtHead,j*p.dtTime,h*p.dtHead+j*p.dtTime,
   dtb+dtc+h*p.dtHead,dtAddress p l i j,
   h*p.rateHead,h*p.biasHead,oub,ouc,oub+ouc,h*p.outHead,j*p.outColumn,
   h*p.outHead+j*p.outColumn,oub+ouc+h*p.outHead,outAddress p l i j]

def Signed32Domain (c : Host) : Prop :=
  ∀ l, c.valid l → ∀ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
    ∀ x ∈ integerTerms c l i.val j.val, x < 2^31

theorem signed32_exact (x : Nat) (hx : x < 2^31) :
    (BitVec.ofNat 32 x).toNat = x ∧ (BitVec.ofNat 32 x).toInt = (x : Int) := by
  have hu : x < 2^32 := by omega
  have hn : (BitVec.ofNat 32 x).toNat = x := by
    exact Nat.mod_eq_of_lt hu
  refine ⟨hn,?_⟩
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

theorem source_indices_exact (c : Host) (domain : Signed32Domain c)
    (l : Launch) (hl : c.valid l) (i : Fin c.layout.BH) (j : Fin c.layout.BC)
    (x : Nat) (hx : x ∈ integerTerms c l i.val j.val) :
    (BitVec.ofNat 32 x).toNat = x ∧ (BitVec.ofNat 32 x).toInt = (x : Int) :=
  signed32_exact x (domain l hl i j x hx)

theorem source_subtraction_exact (c : Host) (l : Launch) (hl : c.valid l) :
    ((c.sequence-l.chunk*c.chunk : Nat) : Int) =
      (c.sequence : Int)-(l.chunk : Int)*(c.chunk : Int) := by
  rw [Int.ofNat_sub (Nat.le_of_lt (chunk_start_lt c l hl)),Nat.cast_mul]

def byteAddress (A : FlatAlloc) (rg : RegionName) (off : Nat) := 4*A.addr rg off
def byteEnd (A : FlatAlloc) (rg : RegionName) := 4*(A.base rg+A.extent rg)

/-- Full allocation ends, not only each access's first byte. -/
def ByteDomain (A : FlatAlloc) : Prop := ∀ rg ∈ A.regions, byteEnd A rg ≤ 2^64

theorem full_byte_access (A : FlatAlloc) (hbyte : ByteDomain A)
    (rg : RegionName) (hr : rg ∈ A.regions) (off : Nat) (ho : off < A.extent rg) :
    byteAddress A rg off % 4 = 0 ∧
    byteAddress A rg off + 4 ≤ byteEnd A rg ∧
    byteAddress A rg off + 3 < 2^64 ∧
    (BitVec.ofNat 64 (byteAddress A rg off)).toNat = byteAddress A rg off ∧
    (BitVec.ofNat 64 (byteAddress A rg off + 3)).toNat = byteAddress A rg off + 3 := by
  have he := hbyte rg hr
  have hfull : byteAddress A rg off + 4 ≤ byteEnd A rg := by
    dsimp [byteAddress,byteEnd,FlatAlloc.addr]
    omega
  have hlast : byteAddress A rg off + 3 < 2^64 := by omega
  refine ⟨by simp [byteAddress],hfull,hlast,?_,?_⟩
  · exact Nat.mod_eq_of_lt (show byteAddress A rg off < 2^64 by omega)
  · exact Nat.mod_eq_of_lt hlast

theorem byte_allocations_disjoint (A : FlatAlloc) (hd : A.Disjoint)
    (r s : RegionName) (hr : r ∈ A.regions) (hs : s ∈ A.regions) (hne : r ≠ s) :
    byteEnd A r ≤ 4*A.base s ∨ byteEnd A s ≤ 4*A.base r := by
  rcases hd r hr s hs hne with h | h
  · exact Or.inl (Nat.mul_le_mul_left 4 h)
  · exact Or.inr (Nat.mul_le_mul_left 4 h)

def Access (c : Host) (r : Regions) (hasBias : Bool) (l : Launch)
    (rg : RegionName) (off : Nat) : Prop :=
  (rg = r.dt ∧ ∃ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
    activeInput c.layout l i.val j.val ∧ off = dtAddress c.layout l i.val j.val) ∨
  (rg = r.rate ∧ ∃ i : Fin c.layout.BH,
    head c.layout l i.val < c.heads ∧ off = head c.layout l i.val*c.rateHead) ∨
  (hasBias = Bool.true ∧ rg = r.bias ∧ ∃ i : Fin c.layout.BH,
    head c.layout l i.val < c.heads ∧ off = head c.layout l i.val*c.biasHead) ∨
  (rg = r.out ∧ ∃ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
    activeOutput c.layout l i.val j.val ∧ off = outAddress c.layout l i.val j.val) ∨
  (rg = r.sum ∧ ∃ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
    activeOutput c.layout l i.val j.val ∧ off = sumAddress c.layout l i.val j.val)

theorem access_bound (A : FlatAlloc) (c : Host) (r : Regions) (hasBias : Bool)
    (hb : AllBounds c r hasBias A) (l : Launch) (hl : c.valid l)
    (rg : RegionName) (off : Nat) (ha : Access c r hasBias l rg off) : off < A.extent rg := by
  rcases ha with ⟨rfl,i,j,hi,rfl⟩ | ⟨rfl,i,hi,rfl⟩ | ⟨hbias,rfl,i,hi,rfl⟩ |
    ⟨rfl,i,j,hi,rfl⟩ | ⟨rfl,i,j,hi,rfl⟩
  · exact (hb l hl).dt i j hi
  · exact (hb l hl).rate i hi
  · exact (hb l hl).bias hbias i hi
  · exact (hb l hl).out i j hi
  · exact (hb l hl).sum i j hi

theorem all_accesses_full_bytes (A : FlatAlloc) (hbyte : ByteDomain A)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (c : Host) (r : Regions) (hasBias : Bool) (hb : AllBounds c r hasBias A)
    (l : Launch) (hl : c.valid l) (rg : RegionName) (off : Nat)
    (ha : Access c r hasBias l rg off) :
    byteAddress A rg off % 4 = 0 ∧
    byteAddress A rg off + 4 ≤ byteEnd A rg ∧ byteAddress A rg off + 3 < 2^64 := by
  have ho := access_bound A c r hasBias hb l hl rg off ha
  have hr : rg ∈ A.regions := by
    by_contra hn
    rw [hcov rg hn] at ho
    omega
  have h := full_byte_access A hbyte rg hr off ho
  exact ⟨h.1,h.2.1,h.2.2.1⟩

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
