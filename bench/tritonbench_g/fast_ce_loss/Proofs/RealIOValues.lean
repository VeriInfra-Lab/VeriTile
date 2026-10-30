import bench.tritonbench_g.fast_ce_loss.Proofs.IOValues
import bench.tritonbench_g.fast_ce_loss.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FastCeLoss.RealIO
open VeriTile.Triton FP.Structural KernelSource PublicIO
noncomputable section

def decode (x : WithBot ℝ) : ℝ := x.unbotD 0

def vector (p : Parameters) (x : Values ℝ .real [p.vocab]) (col : Nat) : ℝ :=
  if h : col<p.vocab then x (⟨col,h⟩,PUnit.unit) else 0

def forwardValues (p : Parameters) (xs : ChannelValues ℝ (forwardInputs p Bool.true)) :
    ChannelValues (WithBot ℝ) (forwardInputs p Bool.true) :=
  Fin.cases (fun j => xs (0 : Fin 3) j)
    (Fin.cases (fun j => some (xs (1 : Fin 3) j))
      (Fin.cases (fun j => some (xs (2 : Fin 3) j)) (fun k => k.elim0)))
def backwardValues (p : Parameters) (xs : ChannelValues ℝ (backwardInputs p Bool.true)) :
    ChannelValues (WithBot ℝ) (backwardInputs p Bool.true) :=
  Fin.cases (fun j => xs (0 : Fin 4) j)
    (Fin.cases (fun j => some (xs (1 : Fin 4) j))
      (Fin.cases (fun j => some (xs (2 : Fin 4) j))
        (Fin.cases (fun j => some (xs (3 : Fin 4) j)) (fun k => k.elim0))))

def domain (p : Parameters) (xs : ChannelValues ℝ (forwardInputs p Bool.true)) : Prop :=
  let target := forwardLabel p Bool.true xs
  target = -100 ∨ (0≤target ∧ target<(p.vocab : Int))

def forwardFormula (p : Parameters) (xs : ChannelValues ℝ (forwardInputs p Bool.true)) :
    ChannelValues ℝ (forwardOutputs p) :=
  Fin.cases (fun _ => KernelMath.fullLSE p (vector p (xs (1 : Fin 3))))
    (Fin.cases (fun _ => KernelMath.rowLoss p (vector p (xs (1 : Fin 3)))
      (forwardLabel p Bool.true xs)) (fun k => k.elim0))
def chunkedFormula (p : Parameters) (chunk : Nat)
    (xs : ChannelValues ℝ (forwardInputs p Bool.true)) : ChannelValues ℝ (forwardOutputs p) :=
  Fin.cases (fun _ => KernelMath.blockLSE p chunk (vector p (xs (1 : Fin 3))))
    (Fin.cases (fun _ => KernelMath.partialLoss p (vector p (xs (1 : Fin 3)))
      (forwardLabel p Bool.true xs)) (fun k => k.elim0))
def backwardFormula (p : Parameters) (chunk : Nat)
    (xs : ChannelValues ℝ (backwardInputs p Bool.true)) : ChannelValues ℝ (backwardOutputs p) :=
  Fin.cases (fun j => KernelReal.gradient p (backwardLabel p Bool.true xs)
    (chunk*p.block+j.1.val) (vector p (xs (1 : Fin 4)) (chunk*p.block+j.1.val))
    (xs (2 : Fin 4) PUnit.unit) (xs (3 : Fin 4) PUnit.unit)) (fun k => k.elim0)

theorem mapCell_id {α : Type} (cell : Cell α) : mapCell id cell=cell := by
  cases cell with | mk d v => cases d <;> rfl

theorem real_cell (cell : Cell (WithBot ℝ)) (raw : WithBot ℝ) (x : ℝ)
    (typed : mapCell id cell=Cell.mk .real raw) (value : cell.read .real=some x) :
    mapCell decode cell=Cell.mk .real x := by
  have eq : cell=Cell.mk .real raw := (mapCell_id cell).symm.trans typed
  rw [eq] at value ⊢
  simp only [Cell.read_same] at value
  rw [value]
  rfl

#specNonCircular forwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular chunkedFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular backwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.FastCeLoss.RealIO
