import bench.tritonbench_g.rowwise_quantization_triton.Proofs.KernelResult
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.Mathematics

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelProfile
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 2000000

/-- One fixed tree, selected without any input payload. Each leaf names a lane;
the public launch proves these are exactly the launched lanes. -/
def reduceTree (ops : Operations α) (tree : Tree) {shape : TileShape}
    (ax : Fin shape.length) : (keepDims : Bool) → Values α .real shape →
      Values α .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false, values => fun out => tree.evaluate ops.maximum (fun i =>
      if hi : i < TileShape.axisDim shape ax then
        values (TileShape.insertAxisIndex shape ax out ⟨i, hi⟩) else ops.zero)
  | Bool.true, values => fun out => tree.evaluate ops.maximum (fun i =>
      if hi : i < TileShape.axisDim shape ax then
        values (TileShape.replaceAxisIndex shape ax out ⟨i, hi⟩) else ops.zero)

/-- Source-used primitives only. Half conversion is one explicitly selected
scalar operation. The rest fallback cannot choose a whole-row result. -/
def algebra (ops : Operations α) (half : α → α) (tree : Tree)
    (rest : Algebra α) : Algebra α :=
  { rest with
    literal := fun _ _ x => if x = 0 then ops.zero else if x = 1 then ops.one
      else if x = 127 then ops.scale else rest.literal none .real x
    binary := fun p d op x y => match op with
      | .mul => ops.multiply x y
      | .div => ops.divide x y
      | _ => rest.binary p d op x y
    unary := fun p op x => match op with
      | .abs _ => ops.abs x
      | _ => rest.unary p op x
    cast := fun p src dst x => match src, dst with
      | .real, .fp16 => half x
      | _, _ => rest.cast p src dst x
    fp32Load := id
    reduceMax := fun _ => reduceTree ops tree }

def equal (ops : Operations α) : α → α → Bool := fun x _ => ops.isZero x

def row {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat → α :=
  fun i => (s.mem p.x (s.pids 0 * p.columns + i)).read .real

theorem maximum_eq {α : Type} [Inhabited α] (ops : Operations α)
    (half : α → α) (rest : Algebra α) (p : Parameters) (launch : Launch)
    (hc : launch.columns = p.columns) (hw : launch.width = p.width) (s : State α) :
    KernelResult.maximum (algebra ops half launch.tree rest) p s =
      rowMaximum ops launch (row p s) := by
  unfold KernelResult.maximum KernelResult.maxInput algebra reduceTree rowMaximum
  simp only [TileShape.axisDim, List.getElem_cons_zero, TileShape.insertAxisIndex]
  apply Tree.evaluate_congr
  intro i _
  by_cases hi : i < p.width
  · simp [hi, KernelResult.offset, rowMaximum, row, hc]
  · have hn : ¬ i < p.columns := by have := launch.covers; omega
    simp [hi, hn, rowMaximum, hc]


theorem scaled_eq {α : Type} [Inhabited α] (ops : Operations α)
    (half : α → α) (rest : Algebra α) (p : Parameters) (launch : Launch)
    (hc : launch.columns = p.columns) (hw : launch.width = p.width) (s : State α)
    (i : TileIndex [p.width]) :
    KernelResult.scaledValue (algebra ops half launch.tree rest) (equal ops) p s i =
      rowScaled ops (row p s i.1.val) (rowMaximum ops launch (row p s)) := by
  unfold KernelResult.scaledValue
  rw [maximum_eq ops half rest p launch hc hw s]
  simp [algebra, equal, rowScaled, row, KernelResult.offset]

theorem stored_eq {α : Type} [Inhabited α] (ops : Operations α)
    (half : α → α) (rest : Algebra α) (p : Parameters) (launch : Launch)
    (hc : launch.columns = p.columns) (hw : launch.width = p.width) (s : State α) :
    KernelResult.maximumStored (algebra ops half launch.tree rest) p s =
      half (rowMaximum ops launch (row p s)) := by
  unfold KernelResult.maximumStored
  rw [maximum_eq ops half rest p launch hc hw s]
  rfl

#axiomsClean maximum_eq
#axiomsClean scaled_eq
#axiomsClean stored_eq
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelProfile
