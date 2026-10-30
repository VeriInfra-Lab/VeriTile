import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.LegacyContext

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 2000000

theorem safeCoordinate_lt (limit coordinate : Nat) (positive : 0<limit) :
    safeCoordinate limit coordinate<limit := by
  unfold safeCoordinate
  split <;> omega

theorem safeCoordinate_active (limit coordinate : Nat) (active : coordinate<limit) :
    safeCoordinate limit coordinate=coordinate := by simp [safeCoordinate,active]

/-- The changed pointer code yields exactly the old observable masked loads.
No arithmetic primitive is equated with another primitive. -/
theorem maskedA_normalization {α : Type} (r : Request) (memory : Nat → α)
    (row k : Nat) (zero : α) :
    (if k<r.inner then memory (safeAOffset r row k) else zero) =
    (if k<r.inner then memory (row*r.strideAM+k*r.strideAK) else zero) := by
  by_cases h : k<r.inner <;> simp [safeAOffset,safeCoordinate,h]

theorem maskedB_normalization {α : Type} (r : Request) (memory : Nat → α)
    (k col : Nat) (zero : α) :
    (if k<r.inner then memory (safeBOffset r k col) else zero) =
    (if k<r.inner then memory (k*r.strideBK+col*r.strideBN) else zero) := by
  by_cases h : k<r.inner <;> simp [safeBOffset,safeCoordinate,h]

theorem store_normalization (r : Request) (row col : Nat)
    (hr : row<r.rows) (hc : col<r.columns) :
    safeCOffset r row col=outputOffset r row col := by
  simp [safeCOffset,outputOffset,safeCoordinate,hr,hc]

theorem mask_normalization (r : Request) (block lane : Nat)
    (hb : block<r.parameters.numKBlocks) :
    (block*r.config.bk+lane<r.inner) ↔ lane<r.inner-block*r.config.bk := by
  have h := (source_subtraction r block hb).1
  omega

theorem safeA_bound {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row k : Nat) (hr : row<r.rows) (hk : 0<r.inner) :
    safeAOffset r row k<t.extentA :=
  t.inputA row hr _ (safeCoordinate_lt _ _ hk)

theorem safeB_bound {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (k col : Nat) (hc : col<r.columns) (hk : 0<r.inner) :
    safeBOffset r k col<t.extentB :=
  t.inputB _ (safeCoordinate_lt _ _ hk) col hc

theorem safeC_bound (r : Request) (row col : Nat) (hr : 0<r.rows) (hc : 0<r.columns) :
    safeCOffset r row col<r.rows*r.columns :=
  output_bound r _ _ (safeCoordinate_lt _ _ hr) (safeCoordinate_lt _ _ hc)

/-- Allocation bounds imply the finite widths, even for unused singleton
strides near INT64_MAX. No padded-envelope or 2^31 premise appears. -/
theorem safeA_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row k : Nat) (hr : row<r.rows) (hk : 0<r.inner) :
    safeAOffset r row k<2^63 ∧ t.baseA+4*safeAOffset r row k<2^64 := by
  have h := safeA_bound r s t row k hr hk
  have ha := t.allocationA
  have hs := t.spanA
  constructor <;> omega

theorem safeB_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (k col : Nat) (hc : col<r.columns) (hk : 0<r.inner) :
    safeBOffset r k col<2^63 ∧ t.baseB+4*safeBOffset r k col<2^64 := by
  have h := safeB_bound r s t k col hc hk
  have ha := t.allocationB
  have hs := t.spanB
  constructor <;> omega

theorem safeC_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row col : Nat) (hr : 0<r.rows) (hc : 0<r.columns) :
    safeCOffset r row col<2^63 ∧ t.baseC+2*safeCOffset r row col<2^64 := by
  have h := safeC_bound r row col hr hc
  have ha := t.allocationC
  have hs := t.spanC
  constructor <;> omega

theorem dimension_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) :
    r.rows<2^62 ∧ r.columns<2^62 ∧ r.inner<2^61 := by
  obtain ⟨hr,hc⟩ := nonempty_program r pid hp
  have hm : r.rows≤r.rows*r.columns := by nlinarith
  have hn : r.columns≤r.rows*r.columns := by nlinarith
  have hk : r.inner≤r.rows*r.inner := by nlinarith
  have ho := t.allocationC
  have ha := t.allocationA
  have hd := t.denseA
  omega

theorem control_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) :
    r.rows+r.config.bm<2^63 ∧ r.columns+r.config.bn<2^63 ∧
    r.inner+r.config.bk<2^63 ∧ programs r<2^63 ∧
    r.config.gm*tileColumns r<2^63 := by
  obtain ⟨hr,hc,hk⟩ := dimension_machine r s t pid hp
  obtain ⟨hmpos,hnpos⟩ := nonempty_program r pid hp
  have hm : tileRows r≤r.rows := by
    cases hh : r.config <;> simp [tileRows,hh,Config.bm] <;> omega
  have hn : tileColumns r≤r.columns := by simp [tileColumns,Config.bn]; omega
  have hg := Nat.mul_le_mul hm hn
  have ha := t.allocationC
  have hgrid : programs r<2^63 := by unfold programs; omega
  cases hh : r.config <;> simp [hh,Config.bm,Config.bn,Config.bk,Config.gm]
  all_goals constructor; omega
  all_goals constructor; omega
  all_goals constructor; omega
  all_goals refine ⟨hgrid,?_⟩
  all_goals simp [tileColumns,Config.bn]; omega

theorem lane_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (i : FPValue.CIndex r.parameters) :
    FPValue.row r.parameters (instanceState r pid s) i.1<2^63 ∧
    FPValue.col r.parameters (instanceState r pid s) i.2.1<2^63 := by
  have hg := grouped_bounds (tileRows r) (tileColumns r) r.config.gm pid r.config.positive.2.2.2 hp
  have hwidth := control_machine r s t pid hp
  have hr : FPValue.row r.parameters (instanceState r pid s) i.1<tileRows r*r.config.bm := by
    change FPValue.pidM r.parameters (instanceState r pid s)*r.config.bm+i.1.val<_
    rw [source_pidM]
    have hi : i.1.val<r.config.bm := i.1.isLt
    have hmul := Nat.mul_le_mul_right r.config.bm (Nat.succ_le_of_lt hg.2.1)
    nlinarith
  have hc : FPValue.col r.parameters (instanceState r pid s) i.2.1<tileColumns r*r.config.bn := by
    change FPValue.pidN r.parameters (instanceState r pid s)*r.config.bn+i.2.1.val<_
    rw [source_pidN]
    have hi : i.2.1.val<r.config.bn := i.2.1.isLt
    have hmul := Nat.mul_le_mul_right r.config.bn (Nat.succ_le_of_lt hg.2.2)
    nlinarith
  have hrows : tileRows r*r.config.bm<r.rows+r.config.bm := by
    cases hh : r.config <;> simp [tileRows,hh,Config.bm] <;> omega
  have hcols : tileColumns r*r.config.bn<r.columns+r.config.bn := by
    simp [tileColumns,Config.bn]; omega
  exact ⟨lt_trans (lt_trans hr hrows) hwidth.1,lt_trans (lt_trans hc hcols) hwidth.2.1⟩

theorem inner_lane_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid block lane : Nat) (hp : pid<programs r) (hb : block<r.parameters.numKBlocks)
    (hl : lane<r.config.bk) : block*r.config.bk+lane<2^63 := by
  have hbnd := (source_subtraction r block hb).1
  have hwidth := (control_machine r s t pid hp).2.2.1
  omega

/-- Check the scalar denominator/subtraction sites, not just final offsets.
Every launched program has positive row/column dimensions and positive grouped
index divisors; the natural subtraction equals signed subtraction. -/
theorem candidate_scalar_control_safe (r : Request) (pid : Nat) (hp : pid<programs r) :
    0<r.rows ∧ 0<r.columns ∧ 0<r.config.gm*tileColumns r ∧
    0<groupSize (tileRows r) (tileColumns r) r.config.gm pid ∧
    groupBase (tileRows r) (tileColumns r) r.config.gm pid<tileRows r ∧
    (tileRows r : Int)-(groupBase (tileRows r) (tileColumns r) r.config.gm pid : Nat)=
      (tileRows r-groupBase (tileRows r) (tileColumns r) r.config.gm pid : Nat) := by
  obtain ⟨hr,hc⟩ := nonempty_program r pid hp
  have hcols : 0<tileColumns r := by
    by_contra hn
    have hz : tileColumns r=0 := by omega
    simp [programs,hz] at hp
  have hgroup := grouped_bounds (tileRows r) (tileColumns r) r.config.gm pid r.config.positive.2.2.2 hp
  have hbase : groupBase (tileRows r) (tileColumns r) r.config.gm pid<tileRows r := by
    have hrow := hgroup.2.1
    unfold tileRow at hrow
    omega
  exact ⟨hr,hc,Nat.mul_pos r.config.positive.2.2.2 hcols,hgroup.1,hbase,by omega⟩

/-- Each multiplication feeding the A pointer, including its implicit byte
scaling, fits signed64. This follows from valid coordinates and storage. -/
theorem a_products_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row k : Nat) (hr : row<r.rows) (hk : 0<r.inner) :
    row*r.strideAM<2^63 ∧ safeCoordinate r.inner k*r.strideAK<2^63 ∧
      4*safeAOffset r row k<2^63 := by
  have h := safeA_bound r s t row k hr hk
  have ha := t.allocationA
  unfold safeAOffset at *
  omega

theorem b_products_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (k col : Nat) (hc : col<r.columns) (hk : 0<r.inner) :
    safeCoordinate r.inner k*r.strideBK<2^63 ∧ col*r.strideBN<2^63 ∧
      4*safeBOffset r k col<2^63 := by
  have h := safeB_bound r s t k col hc hk
  have ha := t.allocationB
  unfold safeBOffset at *
  omega

theorem c_products_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row col : Nat) (hr : 0<r.rows) (hc : 0<r.columns) :
    max r.columns 1*safeCoordinate r.rows row<2^63 ∧
      safeCoordinate r.columns col<2^63 ∧ 2*safeCOffset r row col<2^63 := by
  have h := safeC_bound r row col hr hc
  have ha := t.allocationC
  unfold safeCOffset at *
  omega

#axiomsClean candidate_scalar_control_safe
#axiomsClean a_products_machine
#axiomsClean b_products_machine
#axiomsClean c_products_machine

/-- All actual input pointers (including masked K lanes), all output pointers
(including ragged M/N lanes), and unique active ownership. -/
structure AddressSafety {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) : Prop where
  control : ∀ pid, pid<programs r →
    r.rows+r.config.bm<2^63 ∧ r.columns+r.config.bn<2^63 ∧
    r.inner+r.config.bk<2^63 ∧ programs r<2^63 ∧ r.config.gm*tileColumns r<2^63
  loadA : ∀ row k, row<r.rows → 0<r.inner →
    safeAOffset r row k<t.extentA ∧ safeAOffset r row k<2^63 ∧ t.baseA+4*safeAOffset r row k<2^64
  loadB : ∀ k col, col<r.columns → 0<r.inner →
    safeBOffset r k col<t.extentB ∧ safeBOffset r k col<2^63 ∧ t.baseB+4*safeBOffset r k col<2^64
  store : ∀ row col, 0<r.rows → 0<r.columns →
    safeCOffset r row col<r.rows*r.columns ∧ safeCOffset r row col<2^63 ∧ t.baseC+2*safeCOffset r row col<2^64
  lanes : ∀ pid, pid<programs r → ∀ i : FPValue.CIndex r.parameters,
    FPValue.row r.parameters (instanceState r pid s) i.1<2^63 ∧
    FPValue.col r.parameters (instanceState r pid s) i.2.1<2^63
  innerLane : ∀ pid block lane, pid<programs r → block<r.parameters.numKBlocks →
    lane<r.config.bk → block*r.config.bk+lane<2^63
  uniqueStore : ∀ p q, p<programs r → q<programs r → ∀ i j : FPValue.CIndex r.parameters,
    FPValue.active r.parameters (instanceState r p s) i → FPValue.active r.parameters (instanceState r q s) j →
    FPValue.cOffset r.parameters (instanceState r p s) i=FPValue.cOffset r.parameters (instanceState r q s) j → p=q ∧ i=j

theorem address_safety {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) :
    AddressSafety r s t where
  control := control_machine r s t
  loadA := fun row k hr hk => ⟨safeA_bound r s t row k hr hk,safeA_machine r s t row k hr hk⟩
  loadB := fun k col hc hk => ⟨safeB_bound r s t k col hc hk,safeB_machine r s t k col hc hk⟩
  store := fun row col hr hc => ⟨safeC_bound r row col hr hc,safeC_machine r s t row col hr hc⟩
  lanes := lane_machine r s t
  innerLane := inner_lane_machine r s t
  uniqueStore := source_store_unique r s

#axiomsClean address_safety
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
