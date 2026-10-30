import bench.tritonbench_g.swiglu_fwd.Proofs.WholePublic

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

abbrev signedLimit : Nat := 2^31
abbrev addressLimit : Nat := 2^64

/-- Conservative source index/metadata domain. The 1024 padding budget covers
all six source configurations, including masked lanes. Physical storage views
carry nonwrapping byte placement; different storage identities may overlap.
Index correspondence itself does not require input/output separation. -/
structure FiniteAdmission (p : Prepared α input supplied entry preexisting) (d : Domain p) where
  rows32 : p.batchShape.prod < signedLimit
  inputStride32 : p.flat.rowStride < signedLimit
  outputStride32 : p.out.rowStride < signedLimit
  inputPadded32 : p.batchShape.prod*p.flat.rowStride + (p.ncols+1024) < signedLimit
  outputPadded32 : p.batchShape.prod*p.out.rowStride + (p.ncols+1024) < signedLimit
  byteBase : Nat → Nat
  byteBaseAligned : ∀ allocation, byteBase allocation % 4 = 0
  inputAllocation64 : byteBase p.flat.allocation+4*d.capacity p.flat.allocation ≤ addressLimit
  outputAllocation64 : byteBase p.out.allocation+4*d.capacity p.out.allocation ≤ addressLimit

def row32 (row stride : Nat) : BitVec 32 := BitVec.ofNat 32 row * BitVec.ofNat 32 stride

def start32 (block B : Nat) : BitVec 32 := BitVec.ofNat 32 block * BitVec.ofNat 32 B

def cols32 (block B lane : Nat) : BitVec 32 := start32 block B + BitVec.ofNat 32 lane

/-- Pointer staging and element addition, with signed i32 offsets extended
into the ordinary 64-bit byte pointer arithmetic, and four bytes per FP32. -/
def pointer64 (byteBase viewBase : Nat) (row column : BitVec 32) : BitVec 64 :=
  BitVec.ofNat 64 byteBase + BitVec.ofNat 64 (4*viewBase) +
    BitVec.ofInt 64 row.toInt * BitVec.ofNat 64 4 +
    BitVec.ofInt 64 column.toInt * BitVec.ofNat 64 4

theorem signed_ofNat (n : Nat) (h : n < signedLimit) : (BitVec.ofNat 32 n).toInt = (n : Int) := by
  have hu : n < 2^32 := by dsimp [signedLimit] at h; omega
  have hn : (BitVec.ofNat 32 n).toNat = n := by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt hu]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; dsimp [signedLimit] at h; omega),hn]

theorem config_max (config : Config) : config.block ≤ 1024 := by cases config <;> decide

theorem padded_column (n : Nat) (config : Config) (block lane : Nat)
    (hb : block < (n+config.block-1)/config.block) (hl : lane < config.block) :
    block*config.block+lane < n+1024 := by
  have positive := config.positive
  have maximum := config_max config
  have division := Nat.div_add_mod (n+config.block-1) config.block
  have scaled := Nat.mul_le_mul_left config.block (Nat.succ_le_of_lt hb)
  simp only [Nat.succ_eq_add_one,Nat.mul_add,Nat.mul_one] at scaled
  have truncation : n+config.block-1 ≤ n+config.block := Nat.sub_le _ _
  rw [Nat.mul_comm config.block block] at scaled
  have h : block*config.block+lane < n+config.block := by omega
  omega

structure LaneCorrespondence (p : Prepared α input supplied entry preexisting) (config : Config)
    (r b lane : Nat) : Prop where
  rowInput : (row32 r p.flat.rowStride).toInt = (r*p.flat.rowStride : Nat)
  rowOutput : (row32 r p.out.rowStride).toInt = (r*p.out.rowStride : Nat)
  blockStart : (start32 b config.block).toInt = (b*config.block : Nat)
  column : (cols32 b config.block lane).toInt = (b*config.block+lane : Nat)
  combinedInput : (row32 r p.flat.rowStride + cols32 b config.block lane).toInt =
    (r*p.flat.rowStride+(b*config.block+lane) : Nat)
  combinedOutput : (row32 r p.out.rowStride + cols32 b config.block lane).toInt =
    (r*p.out.rowStride+(b*config.block+lane) : Nat)
  mask : (cols32 b config.block lane).slt (BitVec.ofNat 32 p.ncols) =
    decide (b*config.block+lane < p.ncols)

theorem lane_correspondence (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) (config : Config) (r b lane : Nat)
    (hr : r < p.batchShape.prod) (hb : b < (p.ncols+config.block-1)/config.block)
    (hl : lane < config.block) : LaneCorrespondence p config r b lane := by
  have colBound := padded_column p.ncols config b lane hb hl
  have inputRowBound := Nat.mul_le_mul_right p.flat.rowStride (Nat.le_of_lt hr)
  have outputRowBound := Nat.mul_le_mul_right p.out.rowStride (Nat.le_of_lt hr)
  have inputBound := admitted.inputPadded32
  have outputBound := admitted.outputPadded32
  have hrowx : r*p.flat.rowStride < signedLimit := by omega
  have hrowo : r*p.out.rowStride < signedLimit := by omega
  have hstart : b*config.block < signedLimit := by omega
  have hcol : b*config.block+lane < signedLimit := by omega
  have htotalx : r*p.flat.rowStride+(b*config.block+lane) < signedLimit := by omega
  have htotalo : r*p.out.rowStride+(b*config.block+lane) < signedLimit := by omega
  have hn : p.ncols < signedLimit := by omega
  have colEq : (cols32 b config.block lane).toInt = (b*config.block+lane : Nat) := by
    simp only [cols32,start32,←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact signed_ofNat _ hcol
  refine ⟨?_,?_,?_,colEq,?_,?_,?_⟩
  · simpa only [row32,←BitVec.ofNat_mul] using signed_ofNat _ hrowx
  · simpa only [row32,←BitVec.ofNat_mul] using signed_ofNat _ hrowo
  · simpa only [start32,←BitVec.ofNat_mul] using signed_ofNat _ hstart
  · simpa only [row32,cols32,start32,←BitVec.ofNat_mul,←BitVec.ofNat_add] using signed_ofNat _ htotalx
  · simpa only [row32,cols32,start32,←BitVec.ofNat_mul,←BitVec.ofNat_add] using signed_ofNat _ htotalo
  · change decide ((cols32 b config.block lane).toInt < (BitVec.ofNat 32 p.ncols).toInt) = _
    rw [colEq,signed_ofNat _ hn]
    simp only [Int.ofNat_lt]

theorem pointer64_exact (base viewBase rowValue colValue : Nat) (row column : BitVec 32)
    (hr : row.toInt = (rowValue : Int)) (hc : column.toInt = (colValue : Int))
    (bound : base+4*(viewBase+rowValue+colValue) < addressLimit) :
    (pointer64 base viewBase row column).toNat = base+4*(viewBase+rowValue+colValue) := by
  simp only [pointer64,hr,hc,BitVec.ofInt_natCast,←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  have arithmetic : base+4*viewBase+rowValue*4+colValue*4 = base+4*(viewBase+rowValue+colValue) := by omega
  rw [arithmetic,Nat.mod_eq_of_lt bound]

/-- All active source accesses have the proved 64-bit byte address and stay
inside their original allocation. No access obligation is imposed on an
inactive lane, though its signed column/row arithmetic is still checked. -/
def ActiveAddresses (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) (config : Config) (r b lane : Nat) : Prop :=
  let q := b*config.block+lane
  let column := cols32 b config.block lane
  let xr := row32 r p.flat.rowStride
  let outr := row32 r p.out.rowStride
  (pointer64 (admitted.byteBase p.flat.allocation) p.flat.base xr column).toNat =
    admitted.byteBase p.flat.allocation+4*(p.flat.base+r*p.flat.rowStride+q) ∧
  (pointer64 (admitted.byteBase p.flat.allocation) (p.flat.base+p.ncols) xr column).toNat =
    admitted.byteBase p.flat.allocation+4*(p.flat.base+p.ncols+r*p.flat.rowStride+q) ∧
  (pointer64 (admitted.byteBase p.out.allocation) p.out.base outr column).toNat =
    admitted.byteBase p.out.allocation+4*(p.out.base+r*p.out.rowStride+q) ∧
  p.flat.base+r*p.flat.rowStride+q < d.capacity p.flat.allocation ∧
  p.flat.base+p.ncols+r*p.flat.rowStride+q < d.capacity p.flat.allocation ∧
  p.out.base+r*p.out.rowStride+q < d.capacity p.out.allocation

theorem active_address_correspondence (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) (config : Config) (r b lane : Nat)
    (hr : r < p.batchShape.prod) (hb : b < (p.ncols+config.block-1)/config.block)
    (hl : lane < config.block) (active : b*config.block+lane < p.ncols) :
    ActiveAddresses p d admitted config r b lane := by
  have indices := lane_correspondence p d admitted config r b lane hr hb hl
  have bounds := active_addresses p d r (b*config.block+lane) hr active
  have input64 := admitted.inputAllocation64
  have output64 := admitted.outputAllocation64
  dsimp [ActiveAddresses]
  refine ⟨?_,?_,?_,bounds.1,?_,bounds.2.2⟩
  · apply pointer64_exact _ _ _ _ _ _ indices.rowInput indices.column
    omega
  · apply pointer64_exact _ _ _ _ _ _ indices.rowInput indices.column
    omega
  · apply pointer64_exact _ _ _ _ _ _ indices.rowOutput indices.column
    omega
  · omega

/-- Element offsets are in FP32 cells, so aligned allocation bases yield
aligned X/Y/OUT pointers, including the second chunk's N-element offset. -/
def ActiveAligned (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) (config : Config) (r b lane : Nat) : Prop :=
  (pointer64 (admitted.byteBase p.flat.allocation) p.flat.base
    (row32 r p.flat.rowStride) (cols32 b config.block lane)).toNat % 4 = 0 ∧
  (pointer64 (admitted.byteBase p.flat.allocation) (p.flat.base+p.ncols)
    (row32 r p.flat.rowStride) (cols32 b config.block lane)).toNat % 4 = 0 ∧
  (pointer64 (admitted.byteBase p.out.allocation) p.out.base
    (row32 r p.out.rowStride) (cols32 b config.block lane)).toNat % 4 = 0

theorem active_aligned (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) (config : Config) (r b lane : Nat)
    (addresses : ActiveAddresses p d admitted config r b lane) :
    ActiveAligned p d admitted config r b lane := by
  dsimp [ActiveAddresses] at addresses
  dsimp [ActiveAligned]
  rw [addresses.1,addresses.2.1,addresses.2.2.1]
  have inputAlignment := admitted.byteBaseAligned p.flat.allocation
  have outputAlignment := admitted.byteBaseAligned p.out.allocation
  omega

def IndexCorrespondence (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) : Prop :=
  (BitVec.ofNat 32 p.batchShape.prod).toInt = (p.batchShape.prod : Int) ∧
  (BitVec.ofNat 32 p.ncols).toInt = (p.ncols : Int) ∧
  (BitVec.ofNat 32 p.flat.rowStride).toInt = (p.flat.rowStride : Int) ∧
  (BitVec.ofNat 32 p.out.rowStride).toInt = (p.out.rowStride : Int) ∧
  ∀ config r b lane, r < p.batchShape.prod → b < (p.ncols+config.block-1)/config.block →
    lane < config.block → LaneCorrespondence p config r b lane ∧
    (b*config.block+lane < p.ncols → ActiveAddresses p d admitted config r b lane ∧
      ActiveAligned p d admitted config r b lane)

theorem index_correspondence (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (admitted : FiniteAdmission p d) : IndexCorrespondence p d admitted := by
  have hn : p.ncols < signedLimit := by have h := admitted.inputPadded32; omega
  refine ⟨signed_ofNat _ admitted.rows32,signed_ofNat _ hn,
    signed_ofNat _ admitted.inputStride32,signed_ofNat _ admitted.outputStride32,?_⟩
  intro config r b lane hr hb hl
  refine ⟨lane_correspondence p d admitted config r b lane hr hb hl,?_⟩
  intro active
  have addresses := active_address_correspondence p d admitted config r b lane hr hb hl active
  exact ⟨addresses,active_aligned p d admitted config r b lane addresses⟩

#axiomsClean index_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
