import bench.tritonbench_g.layer_norm_fwd.Proofs.WholeGrid

/-! The eight Python return components, including the saved mean/Rstd,
residual alias, unchanged int64 seeds, and the two views of one mask allocation.
The input and output row strides are independent. All values are read from
the one merged grid memory and compared with independent input-snapshot math.
This is the selected FP32 storage contract in the source Real semantics. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.HostReturn
open VeriTile.Triton
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

/-- Optionality before the kernel launch, directly from `_layer_norm_fwd`.
All selected floating tensor storage is FP32, so residual dtype differences
do not create an extra allocation in this profile. -/
structure Request where
  rms : Bool
  residual : Bool
  bias : Bool
  rowscale : Bool
  x1 : Bool
  w1 : Bool
  b1 : Bool
  returnMask : Bool

def flags (q : Request) (p : ℝ) : Flags where
  rms := q.rms
  residual := q.residual
  residualOut := q.residual || decide (0 < p) || q.rowscale || q.x1
  bias := q.bias
  dropout := decide (0 < p)
  dropoutMask := q.returnMask && decide (0 < p)
  rowscale := q.rowscale
  x1 := q.x1
  w1 := q.w1
  b1 := q.b1

def Admitted (q : Request) (N : Nat) (eps p : ℝ) : Prop :=
  0 < N ∧ N ≤ 16384 ∧ 0 < eps ∧ 0 ≤ p ∧ p < 1 ∧
    (q.x1 = Bool.true → q.rowscale = Bool.false)

abbrev Matrix (M N : Nat) := Fin M → Fin N → MemCell
abbrev Vector (M : Nat) := Fin M → MemCell

@[ext] structure Result (M N seedCount : Nat) where
  y : Matrix M N
  y1 : Option (Matrix M N)
  mean : Option (Vector M)
  rstd : Vector M
  residual : Matrix M N
  seeds : Option (Vector seedCount)
  mask : Option (Matrix M N)
  mask1 : Option (Matrix M N)

def seedCount (f : Flags) (M : Nat) : Nat := if f.x1 then 2 * M else M

/-- Readback of the exact return tuple. The second mask view starts at M*N
in the common boolean allocation; the fallback residual is the input view. -/
def observe (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (s : BlockState) : Result M N (seedCount f M) where
  y := fun i j => s.mem r.y (i.val * stride.y + j.val)
  y1 := if f.w1 then some (fun i j => s.mem r.y1 (i.val * stride.y1 + j.val)) else none
  mean := if f.rms then none else some (fun i => s.mem r.mean i.val)
  rstd := fun i => s.mem r.rstd i.val
  residual := if f.residualOut then
      (fun i j => s.mem r.residualOut (i.val * stride.residualOut + j.val))
    else (fun i j => s.mem r.x (i.val * stride.x + j.val))
  seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
  mask := if f.dropout && f.dropoutMask then
      some (fun i j => s.mem r.dropoutMask (i.val * N + j.val)) else none
  mask1 := if f.dropout && f.dropoutMask && f.x1 then
      some (fun i j => s.mem r.dropoutMask ((M + i.val) * N + j.val)) else none

/-- Independent expected values. Only initial input cells occur here. -/
def oracle (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (s : BlockState) : Result M N (seedCount f M) :=
  let a := snapshot r stride s
  { y := fun i j => MemCell.real (Formula.y f a M N eps p i.val j.val)
    y1 := if f.w1 then some (fun i j => MemCell.real (Formula.y1 f a M N eps p i.val j.val)) else none
    mean := if f.rms then none else some (fun i => MemCell.real (Formula.mean f a M N p i.val))
    rstd := fun i => MemCell.real (Formula.rstd f a M N eps p i.val)
    residual := if f.residualOut then
        (fun i j => MemCell.real (Formula.combined f a M p i.val j.val))
      else (fun i j => s.mem r.x (i.val * stride.x + j.val))
    seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
    mask := if f.dropout && f.dropoutMask then
        some (fun i j => MemCell.of .bool (Formula.keep (a.seeds i.val) p j.val)) else none
    mask1 := if f.dropout && f.dropoutMask && f.x1 then
        some (fun i j => MemCell.of .bool (Formula.keep (a.seeds (M + i.val)) p j.val)) else none }

theorem written_region (f : Flags) (r : Regions) (stride : Strides)
    (M N i : Nat) (region : RegionName) (offset : Nat)
    (h : Contract.written f r stride M N i region offset) :
    Domain.writeRegion f r region := by
  unfold Contract.written at h
  rcases h with h | h | h | h | h | h | h <;> simp_all [Domain.writeRegion]

theorem preserves_input (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState)
    (hsep : Domain.ReadWriteSeparated f r)
    (hp : WholeGrid.Post f r stride M N eps p initial final)
    (region : RegionName) (hr : Domain.readRegion f r region) (offset : Nat) :
    final.mem region offset = initial.mem region offset := by
  apply hp.2 region offset
  intro i _ hw
  exact hsep region hr region (written_region f r stride M N i region offset hw) rfl

theorem y_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (i : Fin M) (j : Fin N) :
    final.mem r.y (i.val * stride.y + j.val) =
      MemCell.real (Formula.y f (snapshot r stride initial) M N eps p i.val j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * stride.y) j.isLt
  have hw : Contract.written f r stride M N i.val r.y (i.val * stride.y + j.val) :=
    Or.inr (Or.inl ⟨rfl, hi⟩)
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem y1_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (hf : f.w1 = Bool.true)
    (i : Fin M) (j : Fin N) :
    final.mem r.y1 (i.val * stride.y1 + j.val) =
      MemCell.real (Formula.y1 f (snapshot r stride initial) M N eps p i.val j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * stride.y1) j.isLt
  have hw : Contract.written f r stride M N i.val r.y1 (i.val * stride.y1 + j.val) :=
    Or.inl ⟨hf, rfl, hi⟩
  rw [hp.1 i.val i.isLt _ _ hw]
  simp [Contract.expectedCell, hf, hi]

theorem rstd_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (i : Fin M) :
    final.mem r.rstd i.val =
      MemCell.real (Formula.rstd f (snapshot r stride initial) M N eps p i.val) := by
  have hw : Contract.written f r stride M N i.val r.rstd i.val := by simp [Contract.written]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem mean_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (hf : f.rms = Bool.false) (i : Fin M) :
    final.mem r.mean i.val =
      MemCell.real (Formula.mean f (snapshot r stride initial) M N p i.val) := by
  have hw : Contract.written f r stride M N i.val r.mean i.val := by simp [Contract.written, hf]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem residual_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (hf : f.residualOut = Bool.true)
    (i : Fin M) (j : Fin N) :
    final.mem r.residualOut (i.val * stride.residualOut + j.val) =
      MemCell.real (Formula.combined f (snapshot r stride initial) M p i.val j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * stride.residualOut) j.isLt
  have hw : Contract.written f r stride M N i.val r.residualOut
      (i.val * stride.residualOut + j.val) := by simp [Contract.written, hf, hi]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem mask_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hp : WholeGrid.Post f r stride M N eps p initial final)
    (hd : f.dropout = Bool.true) (hm : f.dropoutMask = Bool.true) (i : Fin M) (j : Fin N) :
    final.mem r.dropoutMask (i.val * N + j.val) =
      MemCell.of .bool (Formula.keep ((snapshot r stride initial).seeds i.val) p j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * N) j.isLt
  have hsecond : ¬ Contract.inRow ((M + i.val) * N) N (i.val * N + j.val) :=
    fun h => WholeGrid.mask_halves_disjoint i.isLt hi h
  have hw : Contract.written f r stride M N i.val r.dropoutMask (i.val * N + j.val) := by
    simp [Contract.written, hd, hm, hi]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem mask1_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hp : WholeGrid.Post f r stride M N eps p initial final)
    (hd : f.dropout = Bool.true) (hm : f.dropoutMask = Bool.true) (hx : f.x1 = Bool.true)
    (i : Fin M) (j : Fin N) :
    final.mem r.dropoutMask ((M + i.val) * N + j.val) =
      MemCell.of .bool (Formula.keep ((snapshot r stride initial).seeds (M + i.val)) p j.val) := by
  have hi := Contract.inRow_of_col (base := (M + i.val) * N) j.isLt
  have hw : Contract.written f r stride M N i.val r.dropoutMask ((M + i.val) * N + j.val) := by
    simp [Contract.written, hd, hm, hx, hi]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem observe_eq_oracle (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : WholeGrid.Geometry r stride N)
    (hsep : Domain.ReadWriteSeparated f r)
    (hp : WholeGrid.Post f r stride M N eps p initial final) :
    observe f r stride M N final = oracle f r stride M N eps p initial := by
  have hy := funext fun i => funext fun j => y_cell f r stride M N eps p initial final hg hp i j
  have hy1 := fun hf => funext fun i => funext fun j =>
    y1_cell f r stride M N eps p initial final hp hf i j
  have hs := funext fun i => rstd_cell f r stride M N eps p initial final hg hp i
  have hm := fun hf => funext fun i => mean_cell f r stride M N eps p initial final hg hp hf i
  have hr := fun hf => funext fun i => funext fun j =>
    residual_cell f r stride M N eps p initial final hg hp hf i j
  have hx := funext fun (i : Fin M) => funext fun (j : Fin N) =>
    preserves_input f r stride M N eps p initial final hsep hp r.x
      (by simp [Domain.readRegion]) (i.val * stride.x + j.val)
  have hseed := fun (hd : f.dropout = Bool.true) => funext fun (i : Fin (seedCount f M)) =>
    preserves_input f r stride M N eps p initial final hsep hp r.seeds
      (by simp [Domain.readRegion, hd]) i.val
  have hmask := fun hd hm => funext fun i => funext fun j =>
    mask_cell f r stride M N eps p initial final hg hp hd hm i j
  have hmask1 := fun hd hm hx => funext fun i => funext fun j =>
    mask1_cell f r stride M N eps p initial final hg hp hd hm hx i j
  apply Result.ext
  all_goals dsimp only [observe, oracle]
  · exact hy
  · cases hf : f.w1 <;> simp [hf, hy1]
  · cases hf : f.rms <;> simp [hf, hm]
  · exact hs
  · cases hf : f.residualOut <;> simp [hf, hr, hx]
  · cases hf : f.dropout <;> simp [hf, hseed]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> simp [hd, hm, hmask]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1 <;>
      simp [hd, hm, hx, hmask1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.HostReturn
