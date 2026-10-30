import bench.tritonbench_g.triton_softmax.Proofs.KernelReal
import VeriTile.Triton.Memory.KernelSpec.Tile

namespace VeriTile.Bench.Repairs.TritonSoftmax.RowIO
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- The actual selected quote supports regional-to-flat transport. -/
theorem flattenOk (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :
    ((softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [softmax_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

/-- Source trace safety follows the actual load/store expressions. -/
theorem safe (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ i : Fin B, i.val<n → ib+s.pid*ir+i.val*ic<bounds inp)
    (hout : ∀ i : Fin B, i.val<n → ob+s.pid*orow+i.val*oc<bounds out) :
    Kernel.TraceSafe bounds (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_kernel,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.surfaceBody,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,
    stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt]
  exact ⟨fun a ha => hin a ha,fun a ha => hout a ha⟩

/-- Direct source execution reuses the established softmax arithmetic proof;
only the extra named row_max register differs. -/
theorem memory (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (s : BlockState) :
    ∃ t u,
      exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some t ∧
      exec (SoftmaxTriton1.softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some u ∧
      t.mem=u.mem := by
  have terminates : ∃ t,exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some t := by
    simp [exec,softmax_kernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
      Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
      TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
      ComparableDType.lt,hB]
  obtain ⟨t,ht⟩ := terminates
  obtain ⟨u,hu⟩ := SoftmaxTriton1.row_exec out inp ib ob ir ic orow oc n B hB s
  refine ⟨t,u,ht,hu,?_⟩
  simp [exec,softmax_kernel,SoftmaxTriton1.softmax_kernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB] at ht hu
  subst t
  subst u
  refine fold_observe BlockState.mem _ _ _ _ ?_ ?_
  · rfl
  · intro a b i same
    split <;> simp_all [BlockState.writeMem]

/-- Logical row values and exact exterior frame from the actual source. -/
theorem result (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (inj : RowInjective n B oc) (s : BlockState) (xs : Fin B → ℝ)
    (input : ∀ j : Fin B,j.val<n → s.readMem inp (ib+s.pid*ir+j.val*ic)=xs j) :
    ∃ t,exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some t ∧
      (∀ i : Fin B,i.val<n → t.readMem out (ob+s.pid*orow+i.val*oc)=
        Real.exp (xs i)/(∑ j : Fin B,if j.val<n then Real.exp (xs j) else 0)) ∧
      (∀ r a,(∀ i : Fin B,i.val<n → ¬(out=r ∧ ob+s.pid*orow+i.val*oc=a)) → t.mem r a=s.mem r a) := by
  obtain ⟨t,u,ht,hu,hm⟩ := memory out inp ib ob ir ic orow oc n B hB s
  refine ⟨t,ht,?_,?_⟩
  · intro i hi
    have value := SoftmaxTriton1.row_value out inp ib ob ir ic orow oc n B hB inj s u hu i hi
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr n B _ xs input i,
      VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula n B hB xs i hi] at value
    simpa only [BlockState.readMem,hm] using value
  · intro r a miss
    rw [hm]
    exact SoftmaxTriton1.row_frame out inp ib ob ir ic orow oc n B hB s u hu r a miss

#axiomsClean flattenOk
#axiomsClean safe
#axiomsClean result
end
end VeriTile.Bench.Repairs.TritonSoftmax.RowIO
