import bench.repairs.tritonbench_g.softmax_triton1.Kernels
import bench.tritonbench_g.softmax_triton1.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SoftmaxTriton1
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxTriton1 (softmaxSpec softmaxInputTile)

set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- Only active output coordinates must be distinct. This permits singleton
axes with arbitrary strides and does not constrain masked padding addresses. -/
def RowInjective (n B stride : Nat) : Prop :=
  ∀ i j : Fin B, i.val < n → j.val < n → i.val*stride=j.val*stride → i=j

theorem row_exec (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (s : BlockState) :
    ∃ t, exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some t := by
  simp [exec,softmax_kernel,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.surfaceBody,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]

theorem row_value (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (inj : RowInjective n B oc) (s t : BlockState)
    (run : exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some t)
    (i : Fin B) (hi : i.val<n) :
    t.readMem out (ob+s.pid*orow+i.val*oc)=
      softmaxSpec n B (fun j => s.readMem inp (ib+s.pid*ir+j.val*ic)) i := by
  simp [exec,softmax_kernel,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.surfaceBody,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at run
  subst t
  simp only [BlockState.pid_eq]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true (shape := [B]) _
    (fun k => ob+s.pids 0*orow+k.1.val*oc) _ (fun k => k.1.val<n) (i,PUnit.unit)
    (by simpa using hi) (by
      intro k hk eq
      have e : k.1=i := inj k.1 i (by simpa using hk) hi (by simpa using eq)
      cases k with | mk j u => cases u; simp_all)]
  simp [softmaxSpec,softmaxInputTile,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,Tile.bop,Tile.uop,NumericDType.sub,NumericDType.div,hB,hi]
  congr

theorem row_frame (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (s t : BlockState)
    (run : exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s=some t)
    (r : RegionName) (a : Nat)
    (miss : ∀ i : Fin B, i.val<n → ¬(out=r ∧ ob+s.pid*orow+i.val*oc=a)) :
    t.mem r a=s.mem r a := by
  simp [exec,softmax_kernel,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.surfaceBody,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceMax,Tile.reduceMaxDrop,
    Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,hB,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at run
  subst t
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r a _ _ ?_) rfl
  intro i _ hi eq
  exact miss i.1 (by simpa using hi) eq

theorem row_safe (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ i : Fin B, i.val<n → ib+s.pid*ir+i.val*ic<bounds inp)
    (hout : ∀ i : Fin B, i.val<n → ob+s.pid*orow+i.val*oc<bounds out) :
    Kernel.TraceSafe bounds (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_kernel,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,
    stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]
  exact ⟨fun a ha => hin a ha,fun a ha => hout a ha⟩

theorem reciprocal_safe (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ i : Fin B, i.val<n → ib+s.pid*ir+i.val*ic<bounds inp)
    (hout : ∀ i : Fin B, i.val<n → ob+s.pid*orow+i.val*oc<bounds out) :
    Kernel.TraceSafe bounds (reciprocal_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [reciprocal_kernel,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,
    stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]
  exact ⟨fun a ha => hin a ha,fun a ha => hout a ha⟩

theorem reciprocal_flattenOk (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :
    ((reciprocal_kernel out inp ib ob ir ic orow oc n B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [reciprocal_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

theorem row_flattenOk (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :
    ((softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [softmax_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

private theorem mul_recip (a b : Option ℝ) :
    Option.map₂ (· * ·) a (b.map (fun x => x⁻¹)) = Option.map₂ (· / ·) a b := by
  cases a <;> cases b <;> simp [div_eq_mul_inv]

/-- Real projection of the retained reciprocal candidate. -/
theorem reciprocal_exec (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0 < B) (s : BlockState) :
    exec (reciprocal_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s =
      exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp [softmax_kernel,reciprocal_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,exec,stepStmts,stepStmt,
    evalOp.eq_def,BlockState.setReg,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.mul,NumericDType.div,mul_recip,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]

#axiomsClean row_exec
#axiomsClean row_value
#axiomsClean row_frame
#axiomsClean row_safe
end VeriTile.Bench.Repairs.SoftmaxTriton1
