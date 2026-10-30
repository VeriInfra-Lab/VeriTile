import bench.repairs.tritonbench_g.triton_argmax.Proofs.Native

/-! Additive revision002 geometry. The bound is per launch, never a cap on the
number of logical programs. Source launch options explicitly set num_ctas=1.
These lemmas do not rebind historical emitted graphs to the new source. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.SegmentedGrid

def limit : Nat := 65535
def base (segment : Nat) : Nat := segment * limit
def size (total segment : Nat) : Nat := min (total - base segment) limit
def count (total : Nat) : Nat := (total + limit - 1) / limit
def program (segment lane : Nat) : Nat := base segment + lane

/-- Both the driver's signed grid product and its gridX*num_ctas fit int32.
The y/z CUDA limits are trivially met by the one-dimensional launch. -/
theorem legal (total segment : Nat) (active : base segment < total) :
    0 < size total segment ∧ size total segment ≤ 65535 ∧
    size total segment * 1 * 1 < 2^31 ∧ size total segment * 1 < 2^31 := by
  dsimp [size,limit] at *
  omega

private theorem coverage_for (span total p : Nat) (positive : 0 < span) (live : p < total) :
    p / span < (total + span - 1) / span ∧
    p % span < min (total - p / span * span) span ∧ p / span * span + p % span = p := by
  have split := Nat.div_add_mod p span
  rw [Nat.mul_comm span] at split
  have remBound := Nat.mod_lt p positive
  have product := Nat.div_mul_le_self p span
  have quotient : p / span + 1 ≤ (total + span - 1) / span := by
    apply (Nat.le_div_iff_mul_le positive).mpr
    simp only [Nat.add_mul, one_mul]
    omega
  constructor
  · omega
  constructor
  · apply lt_min
    · omega
    · exact remBound
  · simpa [Nat.mul_comm] using split

theorem coverage (total p : Nat) (live : p < total) :
    p / limit < count total ∧ p % limit < size total (p / limit) ∧
    program (p / limit) (p % limit) = p :=
  coverage_for limit total p (by norm_num [limit]) live

theorem no_extra (total segment lane : Nat)
    (inside : lane < size total segment) : program segment lane < total := by
  dsimp [size,program]
  have bound : lane < total - base segment := lt_of_lt_of_le inside (Nat.min_le_left _ _)
  omega

theorem unique (total segment lane segment' lane' : Nat)
    (left : lane < size total segment) (right : lane' < size total segment')
    (same : program segment lane = program segment' lane') :
    segment = segment' ∧ lane = lane' := by
  have hl : lane < limit := lt_of_lt_of_le left (Nat.min_le_right _ _)
  have hr : lane' < limit := lt_of_lt_of_le right (Nat.min_le_right _ _)
  exact Native.output_unique limit segment lane segment' lane' hl hr same

/-- Legal native ctaid.x values satisfy the emitted zext nneg operand condition. -/
theorem lane_nonnegative (total segment lane : Nat)
    (inside : lane < size total segment) : lane < 2^31 := by
  have bound : lane < limit := lt_of_lt_of_le inside (Nat.min_le_right _ _)
  norm_num [limit] at *
  omega

theorem native_program (total segment lane : Nat)
    (inside : lane < size total segment) (extent : total < Native.signedLimit) :
    Native.signed (program segment lane) = Int.ofNat (program segment lane) := by
  exact Native.signed_exact _ ((no_extra _ _ _ inside).trans extent)

/-- Quotient/remainder recovers exactly the prior two-dimensional grid. -/
theorem dimension (rows k p : Nat) (positive : 0 < k) (live : p < rows*k) :
    p/k < rows ∧ p%k < k ∧ (p/k)*k+p%k=p := by
  have remBound := Nat.mod_lt p positive
  have split := Nat.div_add_mod p k
  have rowBound : p/k < rows := (Nat.div_lt_iff_lt_mul positive).mpr live
  exact ⟨rowBound,remBound,by simpa [Nat.mul_comm] using split⟩

theorem dimension_inverse (rows k row inner : Nat)
    (rowLive : row < rows) (innerLive : inner < k) :
    row*k+inner < rows*k ∧ (row*k+inner)/k=row ∧ (row*k+inner)%k=inner := by
  have pos : 0 < k := by omega
  have bound := Native.dim_output_bound rows k row inner rowLive innerLive
  have decoded := dimension rows k (row*k+inner) pos bound
  have pair := Native.output_unique k ((row*k+inner)/k) ((row*k+inner)%k)
    row inner decoded.2.1 innerLive decoded.2.2
  exact ⟨bound,pair.1,pair.2⟩

/-- Every prior row/inner program is reached once, at unchanged coordinates. -/
theorem dimension_coverage (rows k row inner : Nat)
    (rowLive : row < rows) (innerLive : inner < k) :
    ∃ segment lane, lane < size (rows*k) segment ∧
      (program segment lane)/k=row ∧ (program segment lane)%k=inner := by
  have inv := dimension_inverse rows k row inner rowLive innerLive
  have cov := coverage (rows*k) (row*k+inner) inv.1
  refine ⟨(row*k+inner)/limit,(row*k+inner)%limit,cov.2.1,?_,?_⟩
  · rw [cov.2.2]; exact inv.2.1
  · rw [cov.2.2]; exact inv.2.2

#axiomsClean legal
#axiomsClean coverage
#axiomsClean no_extra
#axiomsClean unique
#axiomsClean lane_nonnegative
#axiomsClean native_program
#axiomsClean dimension
#axiomsClean dimension_inverse
#axiomsClean dimension_coverage
end VeriTile.Bench.Repairs.TritonArgmax.SegmentedGrid
