import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.index_select_cat.Proofs.TypedGather
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.IndexSelectCat.LogicalIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section

def index (p : Layout) (rowPid : Nat) (i : Fin p.BI) : Nat := rowPid * p.BI + i.val
def column (p : Layout) (colPid : Nat) (j : Fin p.BC) : Nat := colPid * p.BC + j.val
def Active (p : Layout) (rowPid colPid : Nat) (i : TileIndex [p.BI, p.BC]) : Prop :=
  index p rowPid i.1 < p.rows ∧ column p colPid i.2.1 < p.cols
def read (p : Layout) (colPid : Nat) (indices : Fin p.BI → Int)
    (i : TileIndex [p.BI, p.BC]) : Nat :=
  (indices i.1 * (p.rowStride : Int) + (column p colPid i.2.1 * p.colStride : Nat)).toNat
def write (p : Layout) (rowPid colPid : Nat) (i : TileIndex [p.BI, p.BC]) : Nat :=
  index p rowPid i.1 * p.rowStride + column p colPid i.2.1 * p.colStride
def Unique (p : Layout) (rowPid colPid : Nat) : Prop :=
  ∀ a b, Active p rowPid colPid a → Active p rowPid colPid b →
    write p rowPid colPid a = write p rowPid colPid b → a = b
def initial (rowPid colPid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then rowPid else if ax = 1 then colPid else 0 }

end
end VeriTile.Bench.Repairs.IndexSelectCat.LogicalIO

namespace VeriTile.Bench.Repairs.IndexSelectCat.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section

/-- FP32 gather with an INT64 metadata input. The payload channel denotes
exactly the source cells selected by the signed row indices and column stride;
its values are copied to the explicit destination window. Both tail masks and
all BI/BC dimensions remain visible. There is no private scratch. -/
def gatherIO (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat) : ProgramIO ℝ ℝ where
  inputs := [⟨.int, [p.BI], p.index⟩, ⟨.real, [p.BI, p.BC], p.source⟩]
  outputs := [⟨.real, [p.BI, p.BC], p.output⟩]
  buffers := [p.index, p.source, p.output]
  read := by
    intro port vals
    refine Fin.cases (fun i => LogicalIO.index p rowPid i.1) (fun tail => ?_) port
    have eq : tail = (0 : Fin 1) := Fin.eq_zero tail
    subst tail
    exact LogicalIO.read p colPid (fun i => vals (0 : Fin 2) (i, ()))
  inputMask := by
    intro port vals
    refine Fin.cases (fun i => LogicalIO.index p rowPid i.1 < p.rows) (fun tail => ?_) port
    have eq : tail = (0 : Fin 1) := Fin.eq_zero tail
    subst tail
    exact LogicalIO.Active p rowPid colPid
  write := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact LogicalIO.write p rowPid colPid
  outputMask := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact LogicalIO.Active p rowPid colPid
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port
  schedule := fun _ => [{
    kernel := kernel p
    backend := copyBackend rest
    pids := fun ax => if ax = 0 then rowPid else if ax = 1 then colPid else 0 }]
  encode := id
  observe := fun _ _ => id

/-- The mathematical gather copies each selected source value to its output
coordinate. Selection itself is the metadata-dependent read window in gatherIO. -/
def copyValues {shape : TileShape} (xs : TileIndex shape → ℝ) : TileIndex shape → ℝ := xs

/-- Adapt the independent copy formula to the declared typed channels. -/
def gathered (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat)
    (vals : (gatherIO rest p rowPid colPid).Inputs) :
    (gatherIO rest p rowPid colPid).Outputs := by
  intro port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  exact copyValues (vals (1 : Fin 2))

end
end VeriTile.Bench.Repairs.IndexSelectCat.Correct

namespace VeriTile.Bench.Repairs.IndexSelectCat.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- Signed selection metadata is retained even though the result copies payload. -/
structure GatherValues (p : Layout) where
  indices : Fin p.BI → Int
  payload : Fin p.BI → Fin p.BC → ℝ

def inputCoordinates (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat) :
    (gatherIO rest p rowPid colPid).Inputs ≃ GatherValues p where
  toFun := fun xs => ⟨fun i => xs (0 : Fin 2) (i, ()), fun i j => xs (1 : Fin 2) (i, j, ())⟩
  invFun := fun xs => Fin.cases (fun j => xs.indices j.1)
    (Fin.cases (fun j => xs.payload j.1 j.2.1) (fun i => Fin.elim0 i))
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    · cases j with | mk a b => cases b; rfl
    · cases j with | mk a b => cases b with | mk b c => cases c; rfl
  right_inv := by intro xs; cases xs; rfl

def outputCoordinates (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat) :
    (gatherIO rest p rowPid colPid).Outputs ≃ (Fin p.BI → Fin p.BC → ℝ) where
  toFun := fun xs i j => xs (0 : Fin 1) (i, j, ())
  invFun := fun xs => Fin.cases (fun j => xs j.1 j.2.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    cases j with | mk a b => cases b with | mk b c => cases c; rfl
  right_inv := fun _ => rfl

def logicalGatherIO (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat) :
    LogicalProgramIO ℝ ℝ (GatherValues p) (Fin p.BI → Fin p.BC → ℝ) where
  core := gatherIO rest p rowPid colPid
  inputs := inputCoordinates rest p rowPid colPid
  outputs := outputCoordinates rest p rowPid colPid

end
end VeriTile.Bench.Repairs.IndexSelectCat.Correct
