import bench.tritonbench_g.masked_select.Proofs.Mathematical_WholeExecution

/-! Finite source index/prefix and allocation bridge. Bounds include inactive
lanes; a masked native out_offset of -1 is never interpreted as an access. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
open VeriTile.Triton TypedCopy StableSelection

theorem config_width (c : Config) (h : c ∈ configs) :
    c.width ∈ [256, 512, 1024, 2048, 4096] := by
  obtain ⟨w, _, hm⟩ := List.mem_flatMap.mp h
  obtain ⟨b, hb, heq⟩ := List.mem_map.mp hm
  subst c
  exact hb

theorem config_width_pos (c : Config) (h : c ∈ configs) : 0 < c.width := by
  have hw := config_width c h
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
  rcases hw with hw | hw | hw | hw | hw <;> omega

theorem padded_extent (c : Config) (h : c ∈ configs) (n : Nat) (hn : n < 2^31) :
    grid n c.width * c.width ≤ 2^31 := by
  have hw := config_width c h
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
  rcases hw with hw | hw | hw | hw | hw <;> rw [grid, hw] <;> omega

def index32 (width pid lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 width + BitVec.ofNat 32 lane

def mask32 (n width pid lane : Nat) : Bool :=
  (index32 width pid lane).slt (BitVec.ofNat 32 n)

theorem all_lane_bound (c : Config) (hc : c ∈ configs) (n pid lane : Nat)
    (hn : n < 2^31) (hp : pid < grid n c.width) (hl : lane < c.width) :
    pid * c.width + lane < 2^31 := by
  have he := padded_extent c hc n hn
  have hp' := Nat.mul_le_mul_right c.width (Nat.succ_le_of_lt hp)
  rw [Nat.succ_mul] at hp'
  omega

theorem nat32_exact (n : Nat) (hn : n < 2^31) : (BitVec.ofNat 32 n).toInt = (n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (w := 32) (n := (n : Int)) (by decide)
  · norm_num
  · norm_num; omega

theorem index32_exact (c : Config) (hc : c ∈ configs) (n pid lane : Nat)
    (hn : n < 2^31) (hp : pid < grid n c.width) (hl : lane < c.width) :
    (index32 c.width pid lane).toInt = (pid * c.width + lane : Nat) := by
  simp only [index32, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
  exact nat32_exact _ (all_lane_bound c hc n pid lane hn hp hl)

theorem mask32_exact (c : Config) (hc : c ∈ configs) (n pid lane : Nat)
    (hn : n < 2^31) (hp : pid < grid n c.width) (hl : lane < c.width) :
    mask32 n c.width pid lane = decide (pid * c.width + lane < n) := by
  rw [mask32, BitVec.slt_eq_decide, index32_exact c hc n pid lane hn hp hl, nat32_exact n hn]
  simp only [Int.ofNat_lt]

def prefix64 (mask : Nat → Bool) (i : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (count mask (i + 1))

def destination64 (mask : Nat → Bool) (i : Nat) : BitVec 64 := prefix64 mask i - 1

theorem prefix64_exact (mask : Nat → Bool) (n i : Nat) (hn : n < 2^31) (hi : i < n) :
    (prefix64 mask i).toInt = (count mask (i + 1) : Nat) := by
  have hb := count_le mask (i + 1)
  apply BitVec.toInt_ofInt_eq_self (w := 64) (n := (count mask (i + 1) : Int)) (by decide)
  · norm_num
  · norm_num; omega

theorem destination64_selected (mask : Nat → Bool) (n i : Nat)
    (hn : n < 2^31) (hi : i < n) (hm : mask i = Bool.true) :
    (destination64 mask i).toInt = (count mask i : Nat) ∧ count mask i < count mask n := by
  have hp := inclusive_selected mask hm
  have hb := count_le mask i
  constructor
  · change (BitVec.ofNat 64 (count mask (i + 1)) - BitVec.ofNat 64 1).toInt = _
    rw [BitVec.ofNat_sub_ofNat_of_le _ _ (by decide) (by omega), hp.1]
    apply BitVec.toInt_ofInt_eq_self (w := 64) (n := (count mask i : Int)) (by decide)
    · norm_num
    · norm_num; omega
  · exact selected_strict mask hm hi

/-- A zero prefix yields native -1. It implies the lane is unselected, so
the source's store mask is false; the Nat interpreter's saturation is inert. -/
theorem zero_prefix_masked (mask : Nat → Bool) (i : Nat) (h : count mask (i + 1) = 0) :
    mask i = Bool.false ∧ (destination64 mask i).toInt = -1 := by
  constructor
  · have hh := count_succ mask i
    cases hm : mask i <;> simp_all
  · simp [destination64, prefix64, h]

structure Allocation where
  base : Nat
  length : Nat
  bytes : Nat
  positive : 0 < bytes
  bounded : base + bytes * length ≤ 2^64

def Allocation.address (a : Allocation) (i byte : Nat) : BitVec 64 :=
  BitVec.ofNat 64 a.base + BitVec.ofNat 64 i * BitVec.ofNat 64 a.bytes + BitVec.ofNat 64 byte

theorem allocation_address (a : Allocation) (i byte : Nat)
    (hi : i < a.length) (hb : byte < a.bytes) :
    (a.address i byte).toNat = a.base + i * a.bytes + byte := by
  have hm := Nat.mul_le_mul_right a.bytes (Nat.succ_le_of_lt hi)
  rw [Nat.succ_mul, Nat.mul_comm a.length a.bytes] at hm
  have hbound := a.bounded
  have exactBound : a.base + i * a.bytes + byte < 2^64 := by omega
  simp only [Allocation.address, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat,
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt exactBound]

/-- Every active payload/mask/prefix read lies inside its contiguous producer
window. Only selected lanes access the freshly allocated output window. -/
theorem source_access_bounds (mask : Nat → Bool) (n i : Nat) (hn : n < 2^31)
    (hi : i < n) (input selection cumulative output : Allocation)
    (hinput : input.length = n) (hselection : selection.length = n)
    (hcumulative : cumulative.length = n) (houtput : output.length = count mask n)
    (bi bs bp bo : Nat) (hbi : bi < input.bytes) (hbs : bs < selection.bytes)
    (hbp : bp < cumulative.bytes) (hbo : bo < output.bytes) :
    (input.address i bi).toNat = input.base + i * input.bytes + bi ∧
    (selection.address i bs).toNat = selection.base + i * selection.bytes + bs ∧
    (cumulative.address i bp).toNat = cumulative.base + i * cumulative.bytes + bp ∧
    (mask i = Bool.true →
      (destination64 mask i).toInt = (count mask i : Nat) ∧
      (output.address (count mask i) bo).toNat = output.base + count mask i * output.bytes + bo) := by
  refine ⟨allocation_address input i bi (by omega) hbi,
    allocation_address selection i bs (by omega) hbs,
    allocation_address cumulative i bp (by omega) hbp, ?_⟩
  intro hm
  have hd := destination64_selected mask n i hn hi hm
  exact ⟨hd.1, allocation_address output _ bo (by omega) hbo⟩

def Allocation.Disjoint (a b : Allocation) : Prop :=
  a.base + a.bytes * a.length ≤ b.base ∨ b.base + b.bytes * b.length ≤ a.base

/-- Actual contiguous byte widths, lengths and output freshness from ordinary
Torch allocation contracts. Read buffers may alias each other. -/
structure NativeBuffers (n length : Nat) where
  input : Allocation
  selection : Allocation
  cumulative : Allocation
  output : Allocation
  inputLength : input.length = n
  selectionLength : selection.length = n
  cumulativeLength : cumulative.length = n
  outputLength : output.length = length
  inputWidth : input.bytes = 4
  selectionWidth : selection.bytes = 1
  cumulativeWidth : cumulative.bytes = 8
  outputWidth : output.bytes = 4
  inputAligned : input.base % 4 = 0
  cumulativeAligned : cumulative.base % 8 = 0
  outputAligned : output.base % 4 = 0
  freshInput : output.Disjoint input
  freshSelection : output.Disjoint selection
  freshCumulative : output.Disjoint cumulative

theorem index32_signExtend (c : Config) (hc : c ∈ configs) (n pid lane : Nat)
    (hn : n < 2^31) (hp : pid < grid n c.width) (hl : lane < c.width) :
    (index32 c.width pid lane).signExtend 64 = BitVec.ofNat 64 (pid * c.width + lane) := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide), index32_exact c hc n pid lane hn hp hl]
  symm
  apply BitVec.toInt_ofInt_eq_self (w := 64) (n := (pid * c.width + lane : Nat)) (by decide)
  · exact le_trans (by norm_num) (Int.natCast_nonneg _)
  · have := all_lane_bound c hc n pid lane hn hp hl
    norm_num
    omega

/-- The source's signed Int64 destination word itself, rather than a fresh
mathematical destination, forms the same allocated output byte address. -/
theorem destination_pointer (mask : Nat → Bool) (n i : Nat) (hn : n < 2^31)
    (hi : i < n) (hm : mask i = Bool.true) (output : Allocation)
    (houtput : output.length = count mask n) (byte : Nat) (hb : byte < output.bytes) :
    (BitVec.ofNat 64 output.base + destination64 mask i * BitVec.ofNat 64 output.bytes +
      BitVec.ofNat 64 byte).toNat = output.base + count mask i * output.bytes + byte := by
  have hp := inclusive_selected mask hm
  have hd : destination64 mask i = BitVec.ofNat 64 (count mask i) := by
    change BitVec.ofNat 64 (count mask (i + 1)) - BitVec.ofNat 64 1 = _
    rw [BitVec.ofNat_sub_ofNat_of_le _ _ (by decide) (by omega), hp.1]
  rw [hd]
  exact allocation_address output _ byte
    (by rw [houtput]; exact (destination64_selected mask n i hn hi hm).2) hb

#axiomsClean mask32_exact
#axiomsClean destination64_selected
#axiomsClean source_access_bounds
#axiomsClean destination_pointer
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
