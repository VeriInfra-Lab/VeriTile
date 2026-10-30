import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelScalar
import VeriTile.Triton.Float.ProgramIO

/-! Public logical IO of both complete source grids. FP32 source operations and
storage are unchanged; Real is the independent polynomial interpretation. The
backward gradient input aliases its output and retains the source's addition
of the prepared gradient, including in the native observation. -/
namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.HarmonicIO
open VeriTile.Triton FP.Structural
noncomputable section

def launches {α : Type} (M : Algebra α) (d : Grid.Direction) (p : Parameters) :
    List (Invocation α) :=
  (List.range (Grid.blocks p)).map fun pid => {
    kernel := match d with | .forward => forwardKernel p | .backward => backwardKernel p
    backend := M
    pids := fun _ => pid
    setPids := fun ax => ax == 0 }

def forwardIO {α : Type} (M : Algebra α) (p : Parameters) : ProgramIO α α where
  inputs := [⟨.real, [p.rows, 3], coordinateRegion⟩]
  outputs := [⟨.real, [p.rows, 11], outputRegion⟩]
  buffers := [coordinateRegion, outputRegion]
  read := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.1.val*3+i.2.1.val
  inputMask := fun _ _ _ => True
  write := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.1.val*p.outputWidth+p.columnOffset+i.2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port
  schedule := fun _ => launches M .forward p
  encode := id
  observe := fun _ _ => id

def backwardIO {α : Type} (M : Algebra α) (p : Parameters) : ProgramIO α α where
  inputs := [⟨.real, [p.rows, 3], coordinateRegion⟩,
    ⟨.real, [p.rows, 11], upstreamRegion⟩, ⟨.real, [p.rows, 3], gradientRegion⟩]
  outputs := [⟨.real, [p.rows, 3], gradientRegion⟩]
  buffers := [coordinateRegion, upstreamRegion, gradientRegion]
  read := by
    intro port _
    refine Fin.cases (fun i => i.1.val*3+i.2.1.val) (fun p1 => ?_) port
    refine Fin.cases
      (fun i => i.1.val*p.upstreamRowStride+(p.columnOffset+i.2.1.val)*p.upstreamColStride)
      (fun p2 => ?_) p1
    have eq : p2 = (0 : Fin 1) := Fin.eq_zero p2
    subst p2
    exact fun i => i.1.val*3+i.2.1.val
  inputMask := fun _ _ _ => True
  write := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.1.val*3+i.2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port
  schedule := fun _ => launches M .backward p
  encode := id
  observe := fun _ _ => id

/-- The independent eleven-channel polynomial, evaluated on each coordinate row. -/
def realForward {rows : Nat} (xs : TileIndex [rows, 3] → ℝ) :
    TileIndex [rows, 11] → ℝ := fun i =>
  Polynomial.value i.2.1 (xs (i.1, 0, ())) (xs (i.1, 1, ())) (xs (i.1, 2, ()))

/-- Independent Jacobian transpose times all eleven upstream channels. -/
def realBackward {rows : Nat} (xs : TileIndex [rows, 3] → ℝ)
    (gs : TileIndex [rows, 11] → ℝ) : TileIndex [rows, 3] → ℝ := fun i =>
  Polynomial.vjp (xs (i.1, 0, ())) (xs (i.1, 1, ())) (xs (i.1, 2, ()))
    (fun c => gs (i.1, c, ())) i.2.1

/-- The existing concrete separate-rounding FP32 observation, on every word. -/
def nativeForward {rows : Nat} (xs : TileIndex [rows, 3] → Packing.Word) :
    TileIndex [rows, 11] → Packing.Word := fun i =>
  Packing.word (Software.forward KernelScalar.selectedProfile
    (Packing.value (xs (i.1, 0, ()))) (Packing.value (xs (i.1, 1, ())))
    (Packing.value (xs (i.1, 2, ()))) i.2.1)

/-- The actual prepared gradient is read and added, even for nonzero words. -/
def nativeBackward {rows : Nat} (xs : TileIndex [rows, 3] → Packing.Word)
    (gs : TileIndex [rows, 11] → Packing.Word)
    (old : TileIndex [rows, 3] → Packing.Word) : TileIndex [rows, 3] → Packing.Word := fun i =>
  Packing.word (Software.backward KernelScalar.selectedProfile
    (Packing.value (xs (i.1, 0, ()))) (Packing.value (xs (i.1, 1, ())))
    (Packing.value (xs (i.1, 2, ()))) (fun c => Packing.value (gs (i.1, c, ())))
    (fun c => Packing.value (old (i.1, c, ()))) i.2.1)

def forwardTarget {α : Type} (M : Algebra α) (p : Parameters)
    (f : (TileIndex [p.rows, 3] → α) → TileIndex [p.rows, 11] → α)
    (vals : (forwardIO M p).Inputs) : (forwardIO M p).Outputs := by
  intro port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  exact f (vals (0 : Fin 1))

def backwardTarget {α : Type} (M : Algebra α) (p : Parameters)
    (f : (TileIndex [p.rows, 3] → α) → (TileIndex [p.rows, 11] → α) →
      (TileIndex [p.rows, 3] → α) → TileIndex [p.rows, 3] → α)
    (vals : (backwardIO M p).Inputs) : (backwardIO M p).Outputs := by
  intro port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  exact f (vals (0 : Fin 3)) (vals (1 : Fin 3)) (vals (2 : Fin 3))

def prepared (p : Parameters) (vals : (backwardIO KernelScalar.realBackend p).Inputs) : Prop :=
  ∀ i : TileIndex [p.rows, 3], vals (2 : Fin 3) i = (0 : ℝ)

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.HarmonicIO

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.LogicalIO
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
noncomputable section

abbrev Coordinates (p : Parameters) := Fin p.rows → Fin 3 → ℝ
abbrev Harmonics (p : Parameters) := Fin p.rows → Fin 11 → ℝ

/-- Backward retains all upstream and initial-gradient inputs. -/
structure BackwardInputs (p : Parameters) where
  coordinates : Coordinates p
  upstream : Harmonics p
  initialGradient : Coordinates p

def forwardInputs (p : Parameters) : (forwardIO realBackend p).Inputs ≃ Coordinates p where
  toFun xs row component := xs (0 : Fin 1) (row,component,())
  invFun xs := Fin.cases (fun j => xs j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro xs; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro xs; rfl

def forwardOutputs (p : Parameters) : (forwardIO realBackend p).Outputs ≃ Harmonics p where
  toFun ys row channel := ys (0 : Fin 1) (row,channel,())
  invFun ys := Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro ys; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl

def backwardInputs (p : Parameters) : (backwardIO realBackend p).Inputs ≃ BackwardInputs p where
  toFun xs := ⟨fun r c => xs (0 : Fin 3) (r,c,()),fun r c => xs (1 : Fin 3) (r,c,()),fun r c => xs (2 : Fin 3) (r,c,())⟩
  invFun xs := Fin.cases (fun j => xs.coordinates j.1 j.2.1)
    (Fin.cases (fun j => xs.upstream j.1 j.2.1)
      (Fin.cases (fun j => xs.initialGradient j.1 j.2.1) (fun i => i.elim0)))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def backwardOutputs (p : Parameters) : (backwardIO realBackend p).Outputs ≃ Coordinates p where
  toFun ys row component := ys (0 : Fin 1) (row,component,())
  invFun ys := Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro ys; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl

def Prepared (p : Parameters) (xs : BackwardInputs p) : Prop := ∀ row component,xs.initialGradient row component=0


def forwardView (p : Parameters) : LogicalProgramIO ℝ ℝ (Coordinates p) (Harmonics p) where
  core := forwardIO realBackend p
  inputs := forwardInputs p
  outputs := forwardOutputs p

def backwardView (p : Parameters) : LogicalProgramIO ℝ ℝ (BackwardInputs p) (Coordinates p) where
  core := backwardIO realBackend p
  inputs := backwardInputs p
  outputs := backwardOutputs p

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.LogicalIO
