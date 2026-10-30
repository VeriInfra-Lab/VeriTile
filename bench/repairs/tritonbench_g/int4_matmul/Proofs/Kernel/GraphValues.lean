import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.SourceGraph
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.Complete

namespace VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def observeState (p : Ops α) (s : FP.Structural.State (Node α)) : FP.Structural.State α where
  mem := fun r i => mapCell (observe p) (s.mem r i)
  regs := fun d sh name => (s.regs d sh name).map (fun values j => mapValue (observe p) d (values j))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r i => mapValue (observe p) d (s.undef d r i)

theorem read_observe (p : Ops α) (c : Cell (Node α)) :
    (mapCell (observe p) c).read .real = observe p (c.read .real) := by
  cases c with
  | mk d value => cases d <;> rfl

theorem int_read_observe (p : Ops α) (c : Cell (Node α)) :
    (mapCell (observe p) c).read .int = c.read .int := by
  cases c with
  | mk d value => cases d <;> rfl

theorem consume_value (p : Ops α) (values : List (Group α)) (carry : α) :
    consume p (.value carry) values =
      .value (values.foldl (fun old g => p.dot8 g.1 g.2 old) carry) := by
  induction values generalizing carry with
  | nil => rfl
  | cons g rest ih => exact ih (p.dot8 g.1 g.2 carry)

/-- A completed fused accumulation is a scalar value, with the exact old K8
group order and seed; no zero-padding or arithmetic identity is used. -/
theorem dotTile_value (p : Ops α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (old : Values α .real [m,n]) (i : TileIndex [m,n]) :
    KernelExecution.dotTile (ops p) a b (fun j => .value (old j)) i =
      .value (KernelExecution.dotTile p (fun j => observe p (a j))
        (fun j => observe p (b j)) old i) := by
  rw [←consume_groups,consume_value]
  simp only [groups,List.foldl_map,KernelExecution.dotTile]
  congr 1
  congr 1
  funext carry chunk
  congr 1 <;> funext lane <;> split <;> rfl

theorem difference_observe (p : Ops α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) (k : Nat) (i : KernelExecution.BIndex cfg) :
    KernelExecution.difference cfg (observeState p s) k i = KernelExecution.difference cfg s k i := by
  simp only [KernelExecution.difference,observeState,int_read_observe,
    KernelExecution.innerIndex,KernelExecution.inputInner,KernelExecution.inputColumn,
    KernelExecution.columnBlock,KernelExecution.blocksN,KernelExecution.rowsInGroup,
    KernelExecution.firstRow,KernelExecution.blocksM,KernelExecution.bOffset,KernelExecution.zeroOffset]

theorem aValue_observe (p : Ops α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) (k : Nat) (i : KernelExecution.AIndex cfg) :
    observe p (KernelExecution.aValue (ops p) cfg s k i) =
      KernelExecution.aValue p cfg (observeState p s) k i := by
  unfold KernelExecution.aValue
  change observe p (if _ then _ else _) =
    if KernelExecution.innerIndex cfg s k i.2.1 < cfg.inner then
      (mapCell (observe p) (s.mem cfg.input (KernelExecution.aOffset cfg s k i))).read .real else p.zero
  split <;> simp only [read_observe,ops,observe]

theorem bValue_observe (p : Ops α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) (k : Nat) (i : KernelExecution.BIndex cfg) :
    observe p (KernelExecution.bValue (ops p) cfg s k i) =
      KernelExecution.bValue p cfg (observeState p s) k i := by
  unfold KernelExecution.bValue
  rw [difference_observe]
  change p.mul (p.fromInt _) (observe p (if _ then _ else _)) =
    p.mul (p.fromInt _) (if KernelExecution.innerIndex cfg s k i.1 < cfg.inner then
      (mapCell (observe p) (s.mem cfg.scale (KernelExecution.scaleOffset cfg s k i))).read .real else p.zero)
  split <;> simp only [read_observe,ops,observe]

/-- Projection to the old independent recurrence for arbitrary original
floating inputs and all masked packed-integer loads. -/
theorem accumulation_value (p : Ops α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) (k : Nat) (i : KernelExecution.CIndex cfg) :
    KernelExecution.accumulation (ops p) cfg s k i =
      .value (KernelExecution.accumulation p cfg (observeState p s) k i) := by
  induction k generalizing i with
  | zero => rfl
  | succ k ih =>
    have previous : KernelExecution.accumulation (ops p) cfg s k =
        fun j => .value (KernelExecution.accumulation p cfg (observeState p s) k j) := funext ih
    rw [KernelExecution.accumulation,previous,dotTile_value]
    simp only [aValue_observe,bValue_observe,KernelExecution.accumulation]

#axiomsClean dotTile_value
#axiomsClean accumulation_value
end
end VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
