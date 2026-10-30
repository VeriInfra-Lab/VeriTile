import VeriTile.Triton.Float.InputBindingIO
import bench.tritonbench_g.cache_transform_triton.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.Bounds
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Memory.LogicalIO

/-! Public whole-grid byte-gather IO. The two input channels retain signed
row-table entries and byte payloads; only source output bytes are writable.
Each scheduled invocation is the exact START=0 source with its program ID.
-/
namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- The source invocation shares backend and program-ID bindings with its candidate. -/
abbrev job {α : Type} (M : Algebra α) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := source p
  backend := M
  pids := fun ax => if ax=0 then pid else 0

/-- All programs of the actual source, in the established launch order. -/
def jobs {α : Type} (M : Algebra α) (p : Parameters) (count : Nat) : List (Invocation α) :=
  (List.range count).map (job M p)

/-- Signed-int64 metadata and uint8 payload/output use their exact Structural
integer tags. Capacities are in source region-index units. There is no scratch;
physical byte-pointer transport remains separate from this regional IO. -/
def cacheIO {α : Type} (M : Algebra α) (p : Parameters) (n capacity : Nat) : ProgramIO α α where
  inputs := [⟨.int,[n],p.rows⟩,⟨.nat,[capacity*p.rowBytes],p.cache⟩]
  outputs := [⟨.nat,[p.count],p.output⟩]
  scratch := []
  buffers := [p.rows,p.cache,p.output]
  read := fun _ _ j => TileShape.linearIndex _ j
  inputMask := fun _ _ _ => 0<p.count
  write := fun _ _ j => TileShape.linearIndex _ j
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => jobs M p (blocks p.count)
  encode := id
  observe := fun _ _ => id

namespace CacheIO

def row {α : Type} (M : Algebra α) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) (q : Nat) : Int :=
  if h:q<n then xs (0 : Fin 2) (⟨q,h⟩,PUnit.unit) else 0

def payload {α : Type} (M : Algebra α) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) (o : Nat) : Nat :=
  if h:o<capacity*p.rowBytes then xs (1 : Fin 2) (⟨o,h⟩,PUnit.unit) else 0

/-- Exactly the independent geometry and metadata range conditions formerly
carried by GatherInput. Empty launches do not acquire a metadata requirement. -/
def Domain {α : Type} (M : Algebra α) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) : Prop :=
  0<p.count → p.count=n*p.rowBytes ∧ 0<p.rowBytes ∧
    p.cache≠p.output ∧ p.rows≠p.output ∧ capacity ≤ i64Max ∧
    ∀ q,q<n → 0≤row M p n capacity xs q ∧ (row M p n capacity xs q).toNat<capacity

/-- Independent row-gather formula, preserving all bytes of every output row. -/
def formula {α : Type} (M : Algebra α) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) : (cacheIO M p n capacity).Outputs := by
  intro o
  have ho : o=(0 : Fin 1) := Fin.eq_zero o
  subst o
  exact fun j => payload M p n capacity xs
    ((row M p n capacity xs (j.1.val/p.rowBytes)).toNat*p.rowBytes+j.1.val%p.rowBytes)

end CacheIO
end
end VeriTile.Bench.Repairs.CacheTransform

namespace VeriTile.Bench.Repairs.CacheTransform.PairIO
open VeriTile Triton FP.Structural
noncomputable section
variable {A : Type} [Inhabited A]

def originalJob (M : Algebra A) (p : Parameters) (pid : Nat) : Invocation A :=
  job M p pid

def candidateJob (M : Algebra A) (p : Parameters) (pid : Nat) : Invocation A :=
  { job M p pid with kernel := quotientRemainder p }

def candidate (M : Algebra A) (p : Parameters) (n capacity : Nat) :
    ProgramIO.Variant (cacheIO M p n capacity) where
  schedule := fun _ => (List.range (blocks p.count)).map (candidateJob M p)
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.CacheTransform.PairIO

namespace VeriTile.Bench.Repairs.CacheTransform.CacheIO
open VeriTile.Triton FP.Structural
noncomputable section

structure LogicalInputs (n capacity rowBytes : Nat) where
  rows : Fin n → Int
  bytes : Fin capacity → Fin rowBytes → Nat

def logicalInputsEquiv (M : Algebra ℝ) (p : Parameters) (n capacity : Nat) :
    (cacheIO M p n capacity).Inputs ≃ LogicalInputs n capacity p.rowBytes where
  toFun := fun xs => ⟨fun q => xs (0 : Fin 2) (q,()),
    fun row byte => xs (1 : Fin 2) (finProdFinEquiv (row,byte),())⟩
  invFun := fun xs => Fin.cases (fun j => xs.rows j.1)
    (Fin.cases (fun j => xs.bytes (finProdFinEquiv.symm j.1).1
      (finProdFinEquiv.symm j.1).2) (fun i => Fin.elim0 i))
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    · rcases j with ⟨q,⟨⟩⟩; rfl
    · rcases j with ⟨k,⟨⟩⟩
      change xs (1 : Fin 2) (finProdFinEquiv (finProdFinEquiv.symm k),()) = xs (1 : Fin 2) (k,())
      rw [finProdFinEquiv.apply_symm_apply]
  right_inv := by
    intro xs
    cases xs with
    | mk rows bytes =>
      change LogicalInputs.mk rows (fun row byte =>
        bytes (finProdFinEquiv.symm (finProdFinEquiv (row,byte))).1
          (finProdFinEquiv.symm (finProdFinEquiv (row,byte))).2) = _
      simp only [Equiv.symm_apply_apply]


def logicalOutputsEquiv (M : Algebra ℝ) (p : Parameters) (n capacity : Nat) :
    (cacheIO M p n capacity).Outputs ≃ (Fin p.count → Nat) where
  toFun := fun xs j => xs (0 : Fin 1) (j,())
  invFun := fun f => Fin.cases (fun j => f j.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro xs
    funext o j
    fin_cases o
    rcases j with ⟨j,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

def LogicalInputs.rowAt (xs : LogicalInputs n capacity rowBytes) (row : Nat) : Int :=
  if h : row<n then xs.rows ⟨row,h⟩ else 0

def LogicalInputs.byteAt (xs : LogicalInputs n capacity rowBytes) (row byte : Nat) : Nat :=
  if h : row<capacity ∧ byte<rowBytes then xs.bytes ⟨row,h.1⟩ ⟨byte,h.2⟩ else 0

def logicalDomain (M : Algebra ℝ) (p : Parameters) (n capacity : Nat)
    (xs : LogicalInputs n capacity p.rowBytes) : Prop :=
  Domain M p n capacity ((logicalInputsEquiv M p n capacity).symm xs)

/-- Exact-cell signed metadata and byte tags, including the empty-launch
domain, remain in the original judgment; only tensor coordinates change. -/
def logicalIO (M : Algebra ℝ) (p : Parameters) (n capacity : Nat) :
    LogicalDomainIO (cacheIO M p n capacity).Inputs (cacheIO M p n capacity).Outputs
      (LogicalInputs n capacity p.rowBytes) (Fin p.count → Nat)
      (cacheIO M p n capacity).ImplementsOnCells where
  inputs := logicalInputsEquiv M p n capacity
  outputs := logicalOutputsEquiv M p n capacity

end
end VeriTile.Bench.Repairs.CacheTransform.CacheIO
