import bench.repairs.tritonbench_g.triton_softmax.Proofs.Memory
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.KernelMath
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.SourceContract

namespace VeriTile.Bench.Repairs.TritonSoftmax.RealModel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
noncomputable section
set_option maxHeartbeats 4000000
def access : Access BlockState MemCell ℝ :=
  ⟨BlockState.mem,fun c => match c.readAs .real with | some (some v) => v | _ => 0⟩
abbrev formula := SoftmaxTriton1.RealModel.formula
def launchState (x : Tensor) (p : Nat) (s : BlockState) : BlockState :=
  {s with pids := (fun ax => if ax=0 then p else 0)
          numPids := (fun ax => if ax=0 then x.rows else 1)}

def evaluate (x y : Tensor) (B p : Nat) (s : BlockState) :=
  exec (softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
    y.rowStride y.colStride x.cols B).toAlgKernel (launchState x p s)

theorem source_memory (x y : Tensor) (B p : Nat) (hB : 0<B) (s t : BlockState)
    (runEq : evaluate x y B p s=some t) :
    ∃ u,exec (SoftmaxTriton1.softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride y.rowStride y.colStride x.cols B).toAlgKernel (launchState x p s)=some u ∧ u.mem=t.mem := by
  simp [evaluate,exec,softmax_kernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at runEq
  subst t
  obtain ⟨u,hu⟩ := SoftmaxTriton1.row_exec y.region x.region x.offset y.offset x.rowStride x.colStride y.rowStride y.colStride x.cols B hB (launchState x p s)
  refine ⟨u,hu,?_⟩
  simp [exec,SoftmaxTriton1.softmax_kernel,
    ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at hu
  subst u
  refine fold_observe BlockState.mem _ _ _ _ ?_ ?_
  · rfl
  · intro a b i same
    split <;> simp_all [BlockState.writeMem]

theorem source : SourceContract access evaluate formula := by
  constructor
  intro x y B p s t values shape injective hB cover hp input run
  obtain ⟨u,hu,mem⟩ := source_memory x y B p hB s t run
  have inj : RowInjective x.cols B y.colStride := by
    intro i j hi hj eq
    apply Fin.ext
    exact (injective p i.val p j.val (by rwa [shape.rows]) (by rwa [shape.cols])
      (by rwa [shape.rows]) (by rwa [shape.cols]) (by simp [Tensor.index,eq])).2
  constructor
  · intro c hc
    have cB : c<B := lt_of_lt_of_le hc cover
    let i : Fin B := ⟨c,cB⟩
    have value := row_value y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B hB inj (launchState x p s) u hu i hc
    have same : ∀ j : Fin B, j.val<x.cols →
        (launchState x p s).readMem x.region
          (x.offset+(launchState x p s).pid*x.rowStride+j.val*x.colStride)=values p j.val := by
      intro j hj
      exact input p j.val hp hj
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr x.cols B _
      (fun j => values p j.val) same i] at value
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula x.cols B hB _ i hc] at value
    rw [SoftmaxTriton1.RealModel.padded_sum_eq x.cols B cover (fun j => Real.exp (values p j))] at value
    simpa [access,Access.read,BlockState.readMem,formula,SoftmaxTriton1.RealModel.formula,
      Tensor.index,launchState,BlockState.pid_eq,i,mem] using value
  · intro r a miss
    change t.mem r a=s.mem r a
    rw [←mem]
    exact row_frame y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B hB (launchState x p s) u hu r a (by
        intro i hi eq
        exact miss p (by simp) i.val (by rwa [shape.cols]) eq)

theorem terminates (x y : Tensor) (B p : Nat) (hB : 0<B) (s : BlockState) :
    ∃ t,evaluate x y B p s=some t := by
  simp [evaluate,exec,softmax_kernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]

#specNonCircular formula avoiding [evaluate,softmax_kernel]
#axiomsClean source_memory
#axiomsClean source
#axiomsClean terminates
end
end VeriTile.Bench.Repairs.TritonSoftmax.RealModel
