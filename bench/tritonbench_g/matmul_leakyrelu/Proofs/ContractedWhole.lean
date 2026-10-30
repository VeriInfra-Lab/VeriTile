import bench.tritonbench_g.matmul_leakyrelu.Proofs.FiniteDomain
import bench.tritonbench_g.matmul_leakyrelu.Proofs.FPFusedComplete

/-! Complete observed contracted profile. All geometry and finite-domain facts
are reused from Whole; only the arithmetic graph changes. The incoming live
accumulator is an explicit argument of the dot primitive, never reconstructed
by assuming equality with the source's separate-add graph. -/
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.Contracted
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

structure Numerics (α : Type) where
  zero : α
  dotAcc : (Fin 32 → α) → (Fin 32 → α) → α → α
  nonnegative : α → Bool
  rightSlope : α → α
  castHalf : α → α

def Numerics.activate {α : Type} (Q : Numerics α) (enabled : Bool) (x : α) : α :=
  if enabled then if Q.nonnegative x then x else Q.rightSlope x else x

structure PrimitiveLaws {α : Type} (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) : Prop where
  initialZero : M.literal (some .fp32) .real 0 = Q.zero
  loadZero : M.cast none .real .fp32 (M.literal none .real 0.0) = Q.zero
  dotAcc : ∀ (a b acc : TileIndex [32,32] → α) (i : TileIndex [32,32]),
    D.apply (some .fp32) .fp32 .fp32 .real {} (batch := []) a b acc i =
      Q.dotAcc (fun k => a (i.1,k,PUnit.unit)) (fun k => b (k,i.2.1,PUnit.unit)) (acc i)
  comparison : ∀ x, D.le (some .fp32) .real (M.literal (some .fp32) .real 0.0) x = Q.nonnegative x
  slope : ∀ x, M.binary (some .fp32) .real .mul x
    (M.literal (some .fp32) .real 0.01) = Q.rightSlope x
  halfCast : ∀ x, M.cast none .real .fp16 x = Q.castHalf x

/-- Same padded logical operands, but the actual incoming accumulator enters
one dotAcc primitive. No separate-add/contracted equality is a premise. -/
def blockStep {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col block : Nat) (acc : α) : α :=
  Q.dotAcc (fun k => if block*32+k.val<r.inner then inputA r s row (block*32+k.val) else Q.zero)
    (fun k => if block*32+k.val<r.inner then inputB r s (block*32+k.val) col else Q.zero) acc

def accumulator {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) : α :=
  foldRange (fun block x => blockStep Q r s row col block x) 0 ((r.inner+31)/32) 1 Q.zero

def reference {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) : α :=
  Q.castHalf (Q.activate (r.activation == "leaky_relu") (accumulator Q r s row col))

def Result {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (initial final : State α) : Prop :=
  (∀ row<r.rows, ∀ col<r.columns,
    final.mem r.C (outputOffset r row col)=Cell.mk .fp16 (reference Q r initial row col)) ∧
  (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset)

def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (D : FPFused.DotAcc α)
    (r : Request) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid::rest, s =>
      (FP.Structural.exec (FPFused.backend M D) (contractedKernel r) (instanceState r pid s)).bind
        (runPrograms M D r rest)

def host {α : Type} [Inhabited α] (M : Algebra α) (D : FPFused.DotAcc α)
    (r : Request) (s : State α) : Except HostError (RegionName × State α) :=
  if r.inner ≠ r.innerB then .error .incompatibleDimensions
  else if r.aContiguous = Bool.false then .error .noncontiguousA
  else match runPrograms M D r (List.range (programs r)) s with
    | none => .error .executionFailure
    | some final => .ok (r.C,final)

/-- Definitional geometric identity supplies the existing full finite address,
mask, grid, coverage, and race facts for the contracted profile as well. -/
theorem geometry_shared {α : Type} (r : Request) (s : State α)
    (block : Nat) (i : TileIndex [32,32]) :
    FPFused.aOffset r.contractedParameters s block i=FPValue.aOffset r.parameters s block i ∧
    FPFused.bOffset r.contractedParameters s block i=FPValue.bOffset r.parameters s block i ∧
    FPFused.cOffset r.contractedParameters s i=FPValue.cOffset r.parameters s i ∧
    (FPFused.active r.contractedParameters s i ↔ FPValue.active r.parameters s i) :=
  ⟨rfl,rfl,rfl,Iff.rfl⟩

theorem source_block {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o)
    (i : FPFused.CIndex r.contractedParameters) (hi : FPFused.active r.contractedParameters current i)
    (block : Nat) (acc : FPFused.CIndex r.contractedParameters → α) :
    FPFused.update M D r.contractedParameters .single current block acc i=
      blockStep Q r initial
        (FPFused.row r.contractedParameters current i.1) (FPFused.col r.contractedParameters current i.2.1) block (acc i) := by
  have haRow : FPFused.aRow r.contractedParameters current i.1=FPFused.row r.contractedParameters current i.1 :=
    Nat.mod_eq_of_lt hi.1
  have hbCol : FPFused.bCol r.contractedParameters current i.2.1=FPFused.col r.contractedParameters current i.2.1 :=
    Nat.mod_eq_of_lt hi.2
  have ha (j : Fin 32) :
      FPFused.aValue M r.contractedParameters .single current block (i.1,j,PUnit.unit)=
        (if block*32+j.val<r.inner then
          inputA r initial (FPFused.row r.contractedParameters current i.1) (block*32+j.val) else Q.zero) := by
    have he : FPFused.aOffset r.contractedParameters current block (i.1,j,PUnit.unit)=
        FPFused.row r.contractedParameters current i.1*r.strideAM+(block*32+j.val)*r.strideAK := by
      unfold FPFused.aOffset
      rw [haRow]
      simp only [Request.contractedParameters,FPFused.BK]
      ring
    by_cases hk : block*32+j.val<r.inner
    · have hj : j.val<r.inner-block*32 := by omega
      simp only [FPFused.aValue,Request.contractedParameters,FPFused.BK,hj,if_true]
      change (current.mem r.A (FPFused.aOffset r.contractedParameters current block (i.1,j,PUnit.unit))).read .fp32=_
      rw [he,hA]
      simp [inputA,hk]
      rfl
    · have hj : ¬j.val<r.inner-block*32 := by omega
      simp only [FPFused.aValue,Request.contractedParameters,FPFused.BK,hj,if_false]
      simpa [hk,FPFused.dtype] using law.loadZero
  have hb (j : Fin 32) :
      FPFused.bValue M r.contractedParameters .single current block (j,i.2.1,PUnit.unit)=
        (if block*32+j.val<r.inner then
          inputB r initial (block*32+j.val) (FPFused.col r.contractedParameters current i.2.1) else Q.zero) := by
    have he : FPFused.bOffset r.contractedParameters current block (j,i.2.1,PUnit.unit)=
        (block*32+j.val)*r.strideBK+FPFused.col r.contractedParameters current i.2.1*r.strideBN := by
      unfold FPFused.bOffset
      rw [hbCol]
      simp only [Request.contractedParameters,FPFused.BK]
      ring
    by_cases hk : block*32+j.val<r.inner
    · have hj : j.val<r.inner-block*32 := by omega
      simp only [FPFused.bValue,Request.contractedParameters,FPFused.BK,hj,if_true]
      change (current.mem r.B (FPFused.bOffset r.contractedParameters current block (j,i.2.1,PUnit.unit))).read .fp32=_
      rw [he,hB]
      simp [inputB,hk]
      rfl
    · have hj : ¬j.val<r.inner-block*32 := by omega
      simp only [FPFused.bValue,Request.contractedParameters,FPFused.BK,hj,if_false]
      simpa [hk,FPFused.dtype] using law.loadZero
  unfold FPFused.update
  change D.apply (some .fp32) .fp32 .fp32 .real {} (batch := [])
    (m := 32) (k := 32) (n := 32) _ _ acc i=_
  rw [law.dotAcc]
  unfold blockStep
  apply congrArg₂ (fun a b => Q.dotAcc a b (acc i))
  · funext j; exact ha j
  · funext j; exact hb j

theorem source_reference {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o)
    (i : FPFused.CIndex r.contractedParameters) (hi : FPFused.active r.contractedParameters current i) :
    FPFused.outputValue M D r.contractedParameters .single current i=
      reference Q r initial (FPFused.row r.contractedParameters current i.1) (FPFused.col r.contractedParameters current i.2.1) := by
  have hacc : FPFused.accumulation M D r.contractedParameters .single current i=
      accumulator Q r initial (FPFused.row r.contractedParameters current i.1) (FPFused.col r.contractedParameters current i.2.1) := by
    unfold FPFused.accumulation accumulator
    rw [foldRange_pointwise _ (fun block x => blockStep Q r initial
      (FPFused.row r.contractedParameters current i.1) (FPFused.col r.contractedParameters current i.2.1) block x)
      i (source_block M D Q law r initial current hA hB i hi)]
    rw [law.initialZero]
    congr 1
  unfold FPFused.outputValue reference
  rw [hacc,law.halfCast]
  congr 1
  unfold FPFused.activation Numerics.activate
  simp only [Request.contractedParameters,law.comparison,law.slope]

def Good {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (initial current : State α) (row col : Nat) : Prop :=
  current.mem r.C (outputOffset r row col)=Cell.mk .fp16 (reference Q r initial row col)

theorem source_offset {α : Type} (r : Request) (s : State α) (i : FPFused.CIndex r.contractedParameters) :
    FPFused.cOffset r.contractedParameters s i=
      outputOffset r (FPFused.row r.contractedParameters s i.1) (FPFused.col r.contractedParameters s i.2.1) := by
  simp [FPFused.cOffset,outputOffset,Request.contractedParameters]

theorem program_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α) (pid : Nat)
    (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o) :
    ∃ next, FP.Structural.exec (FPFused.backend M D) (contractedKernel r) (instanceState r pid current)=some next ∧
      (∀ o, next.mem r.A o=initial.mem r.A o) ∧
      (∀ o, next.mem r.B o=initial.mem r.B o) ∧
      (∀ region offset, ¬Footprint r region offset → next.mem region offset=current.mem region offset) ∧
      (∀ row<r.rows, ∀ col<r.columns, Good Q r initial current row col → Good Q r initial next row col) ∧
      (∀ row<r.rows, ∀ col<r.columns, Covered r initial [pid] row col → Good Q r initial next row col) := by
  let start := instanceState r pid current
  obtain ⟨next,he,_,hv,hf⟩ := FPFused.correctness M D r.contractedParameters .single start
    (active_injective r start)
  have value (i : FPFused.CIndex r.contractedParameters) (hi : FPFused.active r.contractedParameters start i) :
      Good Q r initial next (FPFused.row r.contractedParameters start i.1) (FPFused.col r.contractedParameters start i.2.1) := by
    have h := hv i hi
    rw [source_reference M D Q law r initial start hA hB i hi,source_offset] at h
    exact h
  refine ⟨next,he,?_,?_,?_,?_,?_⟩
  · intro o
    exact (hf r.A o (Or.inl (Ne.symm hCA))).trans (hA o)
  · intro o
    exact (hf r.B o (Or.inl (Ne.symm hCB))).trans (hB o)
  · intro region offset hn
    apply hf
    by_cases hr : region=r.C
    · right
      intro i hi heq
      apply hn
      refine ⟨hr,FPFused.row r.contractedParameters start i.1,hi.1,
        FPFused.col r.contractedParameters start i.2.1,hi.2,?_⟩
      exact heq.trans (source_offset r start i)
    · exact Or.inl hr
  · intro row hr col hc hg
    by_cases hw : ∃ i : FPFused.CIndex r.contractedParameters,
        FPFused.active r.contractedParameters start i ∧ outputOffset r row col=FPFused.cOffset r.contractedParameters start i
    · obtain ⟨i,hi,hw⟩ := hw
      rw [source_offset] at hw
      have hh := offset_injective r hc hi.2 hw
      simpa only [hh.1,hh.2] using value i hi
    · have hframe := hf r.C (outputOffset r row col) (Or.inr (by
        intro i hi heq
        exact hw ⟨i,hi,heq⟩))
      exact hframe.trans hg
  · intro row hr col hc hcov
    obtain ⟨p,hp,i,hrow,hcol⟩ := hcov
    have hp' : p=pid := by simpa using hp
    subst p
    have hrow' : FPFused.row r.contractedParameters start i.1=row := hrow
    have hcol' : FPFused.col r.contractedParameters start i.2.1=col := hcol
    have hi : FPFused.active r.contractedParameters start i := ⟨by rw [hrow']; exact hr,by rw [hcol']; exact hc⟩
    simpa only [hrow',hcol'] using value i hi

theorem programs_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α) (ids : List Nat)
    (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o) :
    ∃ final, runPrograms M D r ids current=some final ∧
      (∀ o, final.mem r.A o=initial.mem r.A o) ∧
      (∀ o, final.mem r.B o=initial.mem r.B o) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=current.mem region offset) ∧
      (∀ row<r.rows, ∀ col<r.columns, Good Q r initial current row col → Good Q r initial final row col) ∧
      (∀ row<r.rows, ∀ col<r.columns, Covered r initial ids row col → Good Q r initial final row col) := by
  induction ids generalizing current with
  | nil =>
    refine ⟨current,rfl,hA,hB,fun _ _ _ => rfl,fun _ _ _ _ h => h,?_⟩
    intro row hr col hc hcov
    obtain ⟨p,hp,_⟩ := hcov
    simp at hp
  | cons pid ids ih =>
    obtain ⟨next,he,ha,hb,hframe,hkeep,hvalue⟩ := program_correct M D Q law r initial current pid hCA hCB hA hB
    obtain ⟨final,hfin,ha',hb',hframe',hkeep',hvalue'⟩ := ih next ha hb
    refine ⟨final,?_,ha',hb',?_,?_,?_⟩
    · simp only [runPrograms,he,Option.bind_some,hfin]
    · intro region offset hn
      exact (hframe' region offset hn).trans (hframe region offset hn)
    · intro row hr col hc hg
      exact hkeep' row hr col hc (hkeep row hr col hc hg)
    · intro row hr col hc hcov
      obtain ⟨p,hp,i,hi,hj⟩ := hcov
      rcases List.mem_cons.mp hp with hp | hp
      · subst p
        exact hkeep' row hr col hc (hvalue row hr col hc ⟨pid,by simp,i,hi,hj⟩)
      · exact hvalue' row hr col hc ⟨p,hp,i,hi,hj⟩

theorem grid_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α) (hCA : r.C≠r.A) (hCB : r.C≠r.B) :
    ∃ final, runPrograms M D r (List.range (programs r)) initial=some final ∧ Result Q r initial final := by
  obtain ⟨final,he,_,_,hf,_,hv⟩ := programs_correct M D Q law r initial initial
    (List.range (programs r)) hCA hCB (fun _ => rfl) (fun _ => rfl)
  refine ⟨final,he,?_,hf⟩
  intro row hr col hc
  obtain ⟨pid,hp,i,hi,hj⟩ := cell_covered r initial row col hr hc
  exact hv row hr col hc ⟨pid,List.mem_range.mpr hp,i,hi,hj⟩

theorem host_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α) (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧ Result Q r initial final := by
  obtain ⟨final,he,hp⟩ := grid_correct M D Q law r initial hCA hCB
  exact ⟨final,by simp [host,hdims,hcontiguous,he],hp⟩

theorem host_dimension_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (r : Request) (s : State α)
    (h : r.inner≠r.innerB) : host M D r s=.error .incompatibleDimensions := by
  simp [host,h]

theorem host_contiguity_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (r : Request) (s : State α)
    (hdims : r.inner=r.innerB) (h : r.aContiguous=Bool.false) :
    host M D r s=.error .noncontiguousA := by
  simp [host,hdims,h]

theorem host_empty {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (r : Request) (s : State α)
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true)
    (hempty : r.rows=0 ∨ r.columns=0) : host M D r s=.ok (r.C,s) := by
  rcases hempty with h | h
  all_goals simp [host,hdims,hcontiguous,programs,tileRows,tileColumns,h,runPrograms]



/-- External logical values only; the live accumulator is retained. -/
def logicalBlock {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (block : Nat) (acc : α) : α :=
  Q.dotAcc (fun k => if hk : block*32+k.val<r.inner then A i ⟨block*32+k.val,hk⟩ else Q.zero)
    (fun k => if hk : block*32+k.val<r.inner then B ⟨block*32+k.val,hk⟩ j else Q.zero) acc

def logicalReference {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) : α :=
  let value := foldRange (fun block x => logicalBlock Q r A B i j block x)
    0 ((r.inner+31)/32) 1 Q.zero
  Q.castHalf (Q.activate (r.activation == "leaky_relu") value)

theorem reference_logical {α : Type} [Inhabited α]
    (Q : Numerics α) (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (i : Fin r.rows) (j : Fin r.columns) :
    reference Q r initial i.val j.val=logicalReference Q r A B i j := by
  have hb (block : Nat) (acc : α) : blockStep Q r initial i.val j.val block acc=logicalBlock Q r A B i j block acc := by
    unfold blockStep logicalBlock
    apply congrArg₂ (fun a b => Q.dotAcc a b acc)
    · funext k
      by_cases hk : block*32+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hA i ⟨block*32+k.val,hk⟩
      · simp [hk]
    · funext k
      by_cases hk : block*32+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hB ⟨block*32+k.val,hk⟩ j
      · simp [hk]
  unfold reference accumulator logicalReference
  simp_rw [hb]

theorem logicalReference_zero_inner {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (hk : r.inner=0) :
    logicalReference Q r A B i j=Q.castHalf (Q.activate (r.activation == "leaky_relu") Q.zero) := by
  simp [logicalReference,hk,foldRange]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.Contracted
