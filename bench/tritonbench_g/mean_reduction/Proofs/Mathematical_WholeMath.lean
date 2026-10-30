import Mathlib.Data.List.NodupEquivFin
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Real.Basic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open scoped BigOperators
noncomputable section

/-- The canonical outer-to-inner coordinate linearization. This is independent
of reduction code, tile sizes, accumulators and all floating-point operations. -/
def rowMajor : (n : Nat) → (size : Fin n → Nat) →
    ((i : Fin n) → Fin (size i)) ≃ Fin ((List.ofFn size).prod)
  | 0, _ => {
      toFun := fun _ => ⟨0, by simp⟩
      invFun := fun _ i => Fin.elim0 i
      left_inv := fun _ => Subsingleton.elim _ _
      right_inv := fun y => Fin.ext (by
        have h : y.val < 1 := by simpa using y.isLt
        change 0 = y.val
        omega) }
  | n+1, size => by
      simpa only [List.ofFn_succ,List.prod_cons] using
        (Fin.consEquiv (fun i => Fin (size i))).symm.trans
          ((Equiv.prodCongr (Equiv.refl _) (rowMajor n (fun i => size i.succ))).trans
            finProdFinEquiv)

/-- A logical tensor retains every original axis; strides enter only when its
logical values are read from storage, never in the mathematical mean. -/
abbrev Coordinates {rank : Nat} (shape : Fin rank → Nat) := (i : Fin rank) → Fin (shape i)

abbrev BatchCoordinates {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank)) :=
  (i : {i : Fin rank // i ∉ dims}) → Fin (shape i.val)

abbrev ReductionCoordinates {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank)) :=
  (i : {i : Fin rank // i ∈ dims}) → Fin (shape i.val)

def joinCoordinates {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (b : BatchCoordinates shape dims) (r : ReductionCoordinates shape dims) : Coordinates shape :=
  fun i => if h : i ∈ dims then r ⟨i,h⟩ else b ⟨i,h⟩

/-- Independent selected-axis mean. An explicit empty axis list has a singleton
reduction-coordinate space and therefore is identity; it is not Torch's
special all-axes interpretation of dim=[]. -/
def mathematicalMean {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (x : Coordinates shape → ℝ) (b : BatchCoordinates shape dims) : ℝ :=
  (∑ r : ReductionCoordinates shape dims, x (joinCoordinates shape dims b r)) /
    (Fintype.card (ReductionCoordinates shape dims) : ℝ)

def batchAxes {rank : Nat} (dims : List (Fin rank)) : List (Fin rank) :=
  (List.finRange rank).filter (fun i => decide (i ∉ dims))

def sortedReduction {rank : Nat} (dims : List (Fin rank)) (stride : Fin rank → Nat) :
    List (Fin rank) := dims.mergeSort (fun i j => decide (stride j ≤ stride i))

theorem batchAxes_nodup {rank : Nat} (dims : List (Fin rank)) : (batchAxes dims).Nodup := by
  exact (List.nodup_finRange rank).filter _

theorem batchAxes_membership {rank : Nat} (dims : List (Fin rank)) (i : Fin rank) :
    i ∈ batchAxes dims ↔ i ∉ dims := by simp [batchAxes]

theorem sortedReduction_perm {rank : Nat} (dims : List (Fin rank)) (stride : Fin rank → Nat) :
    (sortedReduction dims stride).Perm dims := List.mergeSort_perm _ _

def axisEnumeration {rank : Nat} (p : Fin rank → Prop) [DecidablePred p]
    (axes : List (Fin rank)) (nodup : axes.Nodup)
    (membership : ∀ i, i ∈ axes ↔ p i) : Fin axes.length ≃ {i : Fin rank // p i} :=
  (nodup.getEquiv axes).trans (Equiv.subtypeEquivRight membership)

/-- The actual List.get ordering is retained; the equivalence is not an
arbitrarily selected enumeration of the coordinate space. -/
def orderedCoordinates {rank : Nat} (shape : Fin rank → Nat) (p : Fin rank → Prop)
    [DecidablePred p] (axes : List (Fin rank)) (nodup : axes.Nodup)
    (membership : ∀ i, i ∈ axes ↔ p i) :
    ((i : {i : Fin rank // p i}) → Fin (shape i.val)) ≃ Fin ((axes.map shape).prod) := by
  let e := axisEnumeration p axes nodup membership
  have size : List.ofFn (fun i : Fin axes.length => shape (e i).val) = axes.map shape := by
    change List.ofFn (fun i : Fin axes.length => shape (axes.get i)) = axes.map shape
    change List.ofFn (shape ∘ axes.get) = axes.map shape
    rw [← List.map_ofFn, List.ofFn_get]
  exact ((Equiv.piCongrLeft (fun i : {i : Fin rank // p i} => Fin (shape i.val)) e).symm.trans
    (rowMajor axes.length (fun i => shape (e i).val))).trans (finCongr (congrArg List.prod size))

def batchLinear {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank)) :
    BatchCoordinates shape dims ≃ Fin (((batchAxes dims).map shape).prod) :=
  orderedCoordinates shape (fun i => i ∉ dims) (batchAxes dims)
    (batchAxes_nodup dims) (batchAxes_membership dims)

def reductionLinear {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (nodup : dims.Nodup) (stride : Fin rank → Nat) :
    ReductionCoordinates shape dims ≃ Fin (((sortedReduction dims stride).map shape).prod) :=
  orderedCoordinates shape (fun i => i ∈ dims) (sortedReduction dims stride)
    ((sortedReduction_perm dims stride).symm.nodup nodup)
    (fun i => (sortedReduction_perm dims stride).mem_iff)

/-- Source constants computed by the shape loops and dim_compress. -/
def rows {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank)) : Nat :=
  ((batchAxes dims).map shape).prod

def columns {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank)) : Nat :=
  (dims.map shape).prod

theorem sorted_columns {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (stride : Fin rank → Nat) :
    ((sortedReduction dims stride).map shape).prod = columns shape dims :=
  ((sortedReduction_perm dims stride).map shape).prod_eq

def reductionLinearSource {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (nodup : dims.Nodup) (stride : Fin rank → Nat) :
    ReductionCoordinates shape dims ≃ Fin (columns shape dims) :=
  (reductionLinear shape dims nodup stride).trans (finCongr (sorted_columns shape dims stride))

def compressedValue {α : Type} {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (nodup : dims.Nodup) (stride : Fin rank → Nat) (x : Coordinates shape → α)
    (row : Fin (rows shape dims)) (col : Fin (columns shape dims)) : α :=
  x (joinCoordinates shape dims ((batchLinear shape dims).symm row)
    ((reductionLinearSource shape dims nodup stride).symm col))

/-- The exact permutation and contiguous linearization of dim_compress retain
the entire independent reduction-coordinate set. No output fact is a premise. -/
theorem compressed_mean_eq_mathematical {rank : Nat} (shape : Fin rank → Nat)
    (dims : List (Fin rank)) (nodup : dims.Nodup) (stride : Fin rank → Nat)
    (x : Coordinates shape → ℝ) (row : Fin (rows shape dims)) :
    (∑ col : Fin (columns shape dims), compressedValue shape dims nodup stride x row col) /
        (columns shape dims : ℝ) =
      mathematicalMean shape dims x ((batchLinear shape dims).symm row) := by
  let e := reductionLinearSource shape dims nodup stride
  have card : Fintype.card (ReductionCoordinates shape dims) = columns shape dims :=
    (Fintype.card_congr e).trans (Fintype.card_fin _)
  unfold mathematicalMean compressedValue
  rw [card]
  congr 1
  exact e.symm.sum_comp (fun r => x (joinCoordinates shape dims ((batchLinear shape dims).symm row) r))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
