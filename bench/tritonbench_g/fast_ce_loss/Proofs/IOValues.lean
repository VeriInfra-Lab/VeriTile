import bench.tritonbench_g.fast_ce_loss.Proofs.IOLayout
import bench.tritonbench_g.fast_ce_loss.Proofs.LogicalModel

namespace VeriTile.Bench.Repairs.FastCeLoss.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

def tile (p : Parameters) (wholeRow : Bool) (chunk : Nat)
    (x : Values α .real [width p wholeRow]) (lane : Fin p.block) : α :=
  let index := if wholeRow then chunk*p.block+lane.val else lane.val
  if h : index < width p wholeRow then x (⟨index,h⟩,PUnit.unit) else default

theorem tile_index (p : Parameters) (wholeRow : Bool) (chunk : Nat)
    (lane : Fin p.block) (active : chunk*p.block+lane.val<p.vocab) :
    (if wholeRow then chunk*p.block+lane.val else lane.val) < width p wholeRow := by
  cases wholeRow <;> simp only [width,Bool.false_eq_true,if_false,if_true]
  · exact lane.isLt
  · exact active

def nativeDomain (p : Parameters) (wholeRow : Bool)
    (xs : ChannelValues α (forwardInputs p wholeRow)) : Prop :=
  forwardLabel p wholeRow xs = -100 ∨ 0≤forwardLabel p wholeRow xs

def forwardFormula (M : Algebra α) (p : Parameters) (wholeRow : Bool)
    (xs : ChannelValues α (forwardInputs p wholeRow)) : ChannelValues α (forwardOutputs p) :=
  Fin.cases (fun _ => KernelModel.cast M (LogicalModel.lse M p 0
      (tile (α:=α) p wholeRow 0 (xs (1 : Fin 3)))))
    (Fin.cases (fun _ => KernelModel.cast M (LogicalModel.forwardLoss M p
      (tile (α:=α) p wholeRow 0 (xs (1 : Fin 3))) (forwardLabel p wholeRow xs)
      (xs (2 : Fin 3) PUnit.unit))) (fun k => k.elim0))

def chunkedFormula (M : Algebra α) (p : Parameters) (chunk : Nat) (wholeRow : Bool)
    (xs : ChannelValues α (forwardInputs p wholeRow)) : ChannelValues α (forwardOutputs p) :=
  Fin.cases (fun _ => KernelModel.cast M (LogicalModel.lse M p chunk
      (tile (α:=α) p wholeRow chunk (xs (1 : Fin 3)))))
    (Fin.cases (fun _ => KernelModel.cast M (LogicalModel.chunkLoss M p
      (forwardLabel p wholeRow xs) (xs (2 : Fin 3) PUnit.unit))) (fun k => k.elim0))

def backwardFormula (M : Algebra α) (p : Parameters) (chunk : Nat) (wholeRow : Bool)
    (xs : ChannelValues α (backwardInputs p wholeRow)) : ChannelValues α (backwardOutputs p) :=
  Fin.cases (fun j => LogicalModel.gradient M p (chunk*p.block+j.1.val)
    (backwardLabel p wholeRow xs) (tile (α:=α) p wholeRow chunk (xs (1 : Fin 4)) j.1)
    (xs (2 : Fin 4) PUnit.unit) (xs (3 : Fin 4) PUnit.unit)) (fun k => k.elim0)

#specNonCircular forwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,exec,run]
#specNonCircular chunkedFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,exec,run]
#specNonCircular backwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,exec,run]
end
end VeriTile.Bench.Repairs.FastCeLoss.PublicIO
