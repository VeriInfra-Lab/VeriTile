import bench.tritonbench_g.swiglu_backward.Proofs.WholePublic
import bench.tritonbench_g.swiglu_fwd.Proofs.WholeIndex

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

inductive Access where
  | x | y | dout | dx | dy | out
  deriving DecidableEq, Repr

def Access.view (a : Access) (p : Prepared α call entry preexisting) : Matrix :=
  match a with
  | .x => p.xy.matrix
  | .y => {p.xy.matrix with base := p.xy.matrix.base+p.ncols}
  | .dout => p.dout.matrix
  | .dx => p.view .dx
  | .dy => p.view .dy
  | .out => p.view .out

def Access.active (a : Access) (R : Bool) : Prop := a ≠ .out ∨ R = Bool.true

abbrev signedLimit : Nat := 2^31
abbrev addressLimit : Nat := 2^64
abbrev row32 := SwigluFwd.Whole.row32
abbrev start32 := SwigluFwd.Whole.start32
abbrev cols32 := SwigluFwd.Whole.cols32
abbrev pointer64 := SwigluFwd.Whole.pointer64

theorem access_bound (p : Prepared α call entry preexisting) (d : Domain p)
    (a : Access) (on : a.active call.recompute) (r q : Nat)
    (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    (a.view p).base+r*(a.view p).rowStride+q < d.capacity (a.view p).allocation := by
  cases a with
  | x => simpa [Access.view,SwigluFwd.Whole.Matrix.address,p.xy.unit_offset r q (by rwa [p.inputRows]) (by rw [p.inputCols]; omega)] using
      d.inputBounds r q hr (by omega)
  | y => simpa [Access.view,SwigluFwd.Whole.Matrix.address,p.xy.unit_offset r (q+p.ncols) (by rwa [p.inputRows]) (by rw [p.inputCols]; omega),
      Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using d.inputBounds r (p.ncols+q) hr (by omega)
  | dout => simpa [Access.view,SwigluFwd.Whole.Matrix.address,p.dout.unit_offset r q (by rwa [d.upstreamShape.1]) (by rwa [d.upstreamShape.2])] using
      d.upstreamBounds r q hr hq
  | dx => exact d.outputBounds .dx (Or.inl (by decide)) r q hr hq
  | dy => exact d.outputBounds .dy (Or.inl (by decide)) r q hr hq
  | out =>
      have on' : call.recompute = Bool.true := by simpa [Access.active] using on
      exact d.outputBounds .out (Or.inr on') r q hr hq

/-- Finite ABI and allocation conditions for all six views, all six tuning
configurations and all masked padding. Bases and byte capacities refer to
physical allocations; X/Y and DX/DY intentionally share those allocations. -/
structure FiniteAdmission (p : Prepared α call entry preexisting) (d : Domain p) where
  rows32 : p.batchShape.prod < signedLimit
  columns32 : p.ncols < signedLimit
  stride32 : ∀ a : Access, a.active call.recompute → (a.view p).rowStride < signedLimit
  padded32 : ∀ a : Access, a.active call.recompute →
    p.batchShape.prod*(a.view p).rowStride+(p.ncols+1024) < signedLimit
  byteBase : Nat → Nat
  aligned : ∀ allocation, byteBase allocation % 4 = 0
  allocation64 : ∀ allocation, byteBase allocation+4*d.capacity allocation ≤ addressLimit
  padded64 : ∀ a : Access, a.active call.recompute →
    byteBase (a.view p).allocation+4*((a.view p).base+
      p.batchShape.prod*(a.view p).rowStride+(p.ncols+1024)) ≤ addressLimit
  allocationsDisjoint : ∀ a b, a ≠ b →
    byteBase a+4*d.capacity a ≤ byteBase b ∨ byteBase b+4*d.capacity b ≤ byteBase a

theorem config_max (config : Config) : config.block ≤ 1024 := by cases config <;> decide

theorem padded_column (n : Nat) (config : Config) (b lane : Nat)
    (hb : b < (n+config.block-1)/config.block) (hl : lane < config.block) :
    b*config.block+lane < n+1024 := by
  have positive := config.positive
  have maximum := config_max config
  have division := Nat.div_add_mod (n+config.block-1) config.block
  have scaled := Nat.mul_le_mul_left config.block (Nat.succ_le_of_lt hb)
  simp only [Nat.succ_eq_add_one,Nat.mul_add,Nat.mul_one] at scaled
  have truncation : n+config.block-1 ≤ n+config.block := Nat.sub_le _ _
  rw [Nat.mul_comm config.block b] at scaled
  have h : b*config.block+lane < n+config.block := by omega
  omega

structure LaneCorrespondence (p : Prepared α call entry preexisting) (a : Access)
    (config : Config) (r b lane : Nat) : Prop where
  row : (row32 r (a.view p).rowStride).toInt = (r*(a.view p).rowStride : Nat)
  start : (start32 b config.block).toInt = (b*config.block : Nat)
  column : (cols32 b config.block lane).toInt = (b*config.block+lane : Nat)
  combined : (row32 r (a.view p).rowStride+cols32 b config.block lane).toInt =
    (r*(a.view p).rowStride+(b*config.block+lane) : Nat)
  mask : (cols32 b config.block lane).slt (BitVec.ofNat 32 p.ncols) =
    decide (b*config.block+lane < p.ncols)

theorem lane_correspondence (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) (a : Access) (on : a.active call.recompute)
    (config : Config) (r b lane : Nat) (hr : r < p.batchShape.prod)
    (hb : b < (p.ncols+config.block-1)/config.block) (hl : lane < config.block) :
    LaneCorrespondence p a config r b lane := by
  have columnBound := padded_column p.ncols config b lane hb hl
  have rowBound := Nat.mul_le_mul_right (a.view p).rowStride (Nat.le_of_lt hr)
  have total := finite.padded32 a on
  have hrow : r*(a.view p).rowStride < signedLimit := by omega
  have hstart : b*config.block < signedLimit := by omega
  have hcol : b*config.block+lane < signedLimit := by omega
  have htotal : r*(a.view p).rowStride+(b*config.block+lane) < signedLimit := by omega
  have hc : (cols32 b config.block lane).toInt = (b*config.block+lane : Nat) := by
    simp only [cols32,start32,SwigluFwd.Whole.cols32,SwigluFwd.Whole.start32,
      ←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact SwigluFwd.Whole.signed_ofNat _ hcol
  refine ⟨?_,?_,hc,?_,?_⟩
  · simpa only [row32,SwigluFwd.Whole.row32,←BitVec.ofNat_mul] using SwigluFwd.Whole.signed_ofNat _ hrow
  · simpa only [start32,SwigluFwd.Whole.start32,←BitVec.ofNat_mul] using SwigluFwd.Whole.signed_ofNat _ hstart
  · simpa only [row32,cols32,start32,SwigluFwd.Whole.row32,SwigluFwd.Whole.cols32,
      SwigluFwd.Whole.start32,←BitVec.ofNat_mul,←BitVec.ofNat_add] using SwigluFwd.Whole.signed_ofNat _ htotal
  · change decide ((cols32 b config.block lane).toInt < (BitVec.ofNat 32 p.ncols).toInt) = _
    rw [hc,SwigluFwd.Whole.signed_ofNat _ finite.columns32]
    simp only [Int.ofNat_lt]

def address (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) (a : Access) (config : Config) (r b lane : Nat) : BitVec 64 :=
  pointer64 (finite.byteBase (a.view p).allocation) (a.view p).base
    (row32 r (a.view p).rowStride) (cols32 b config.block lane)

def ByteSpan (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) (a : Access) (config : Config) (r b lane : Nat) : Prop :=
  let addr := (address p d finite a config r b lane).toNat
  addr % 4 = 0 ∧
  ∀ byte, byte < 4 → finite.byteBase (a.view p).allocation ≤ addr+byte ∧
    addr+byte < finite.byteBase (a.view p).allocation+4*d.capacity (a.view p).allocation ∧
    addr+byte < addressLimit

theorem formed_pointer (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) (a : Access) (on : a.active call.recompute)
    (config : Config) (r b lane : Nat) (hr : r < p.batchShape.prod)
    (hb : b < (p.ncols+config.block-1)/config.block) (hl : lane < config.block) :
    (address p d finite a config r b lane).toNat = finite.byteBase (a.view p).allocation+
      4*((a.view p).base+r*(a.view p).rowStride+(b*config.block+lane)) := by
  have index := lane_correspondence p d finite a on config r b lane hr hb hl
  have col := padded_column p.ncols config b lane hb hl
  have row := Nat.mul_le_mul_right (a.view p).rowStride (Nat.le_of_lt hr)
  apply SwigluFwd.Whole.pointer64_exact _ _ _ _ _ _ index.row index.column
  have limit := finite.padded64 a on
  dsimp [addressLimit,SwigluFwd.Whole.addressLimit] at limit ⊢
  omega

theorem full_byte_span (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) (a : Access) (on : a.active call.recompute)
    (config : Config) (r b lane : Nat) (hr : r < p.batchShape.prod)
    (hb : b < (p.ncols+config.block-1)/config.block) (hl : lane < config.block)
    (active : b*config.block+lane < p.ncols) : ByteSpan p d finite a config r b lane := by
  have pointer := formed_pointer p d finite a on config r b lane hr hb hl
  have bounds := access_bound p d a on r (b*config.block+lane) hr active
  have alignment := finite.aligned (a.view p).allocation
  have limit := finite.allocation64 (a.view p).allocation
  dsimp [ByteSpan]
  rw [pointer]
  constructor
  · omega
  · intro byte hbyte
    omega

/-- Complete four-byte accesses to distinct aligned cells cannot overlap,
including DX/DY chunks in one backing allocation. -/
theorem distinct_cell_byte_separation (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) (a b x y : Nat)
    (hx : x < d.capacity a) (hy : y < d.capacity b) (different : (a,x) ≠ (b,y))
    (i j : Nat) (hi : i < 4) (hj : j < 4) :
    finite.byteBase a+4*x+i ≠ finite.byteBase b+4*y+j := by
  by_cases same : a = b
  · subst b
    have off : x ≠ y := fun h => different (by simp [h])
    omega
  · rcases finite.allocationsDisjoint a b same with left | right <;> omega

def IndexCorrespondence (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) : Prop :=
  (BitVec.ofNat 32 p.batchShape.prod).toInt = (p.batchShape.prod : Int) ∧
  (BitVec.ofNat 32 p.ncols).toInt = (p.ncols : Int) ∧
  (∀ a : Access, a.active call.recompute →
    (BitVec.ofNat 32 (a.view p).rowStride).toInt = ((a.view p).rowStride : Int)) ∧
  (∀ a : Access, a.active call.recompute → ∀ config r b lane,
    r < p.batchShape.prod → b < (p.ncols+config.block-1)/config.block → lane < config.block →
    LaneCorrespondence p a config r b lane ∧
    (address p d finite a config r b lane).toNat = finite.byteBase (a.view p).allocation+
      4*((a.view p).base+r*(a.view p).rowStride+(b*config.block+lane)) ∧
    (b*config.block+lane < p.ncols → ByteSpan p d finite a config r b lane)) ∧
  (∀ a b x y, x < d.capacity a → y < d.capacity b → (a,x) ≠ (b,y) →
    ∀ i j, i < 4 → j < 4 → finite.byteBase a+4*x+i ≠ finite.byteBase b+4*y+j)

theorem index_correspondence (p : Prepared α call entry preexisting) (d : Domain p)
    (finite : FiniteAdmission p d) : IndexCorrespondence p d finite := by
  refine ⟨SwigluFwd.Whole.signed_ofNat _ finite.rows32,
    SwigluFwd.Whole.signed_ofNat _ finite.columns32,?_,?_,?_⟩
  · intro a on; exact SwigluFwd.Whole.signed_ofNat _ (finite.stride32 a on)
  · intro a on config r b lane hr hb hl
    exact ⟨lane_correspondence p d finite a on config r b lane hr hb hl,
      formed_pointer p d finite a on config r b lane hr hb hl,
      full_byte_span p d finite a on config r b lane hr hb hl⟩
  · exact distinct_cell_byte_separation p d finite

#axiomsClean index_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
