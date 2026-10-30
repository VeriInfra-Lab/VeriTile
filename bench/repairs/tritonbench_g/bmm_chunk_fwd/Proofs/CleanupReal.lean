import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.CleanupSource
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Cleanup
namespace VeriTile.Bench.Repairs.BmmChunkFwd.CleanupReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def kernel (h : Host.Call) : Kernel :=
  (Kernels.clearUpper h.Out h.shape.chunk (h.shape.chunks*h.shape.groups)).eraseDType.toAlgKernel

def store (out : RegionName) (memory : RegionName → Nat → MemCell) (address : Nat) :
    RegionName → Nat → MemCell :=
  fun r off => if r=out ∧ off=address then MemCell.real 0 else memory r off

theorem fold_memory {ι : Type} (out : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (l : List ι) (s : BlockState) :
    (l.foldl (fun t i => if active i then t.writeMem out (off i) 0 else t) s).mem=
      (l.filterMap (fun i => if active i then some (off i) else none)).foldl (store out) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,List.filterMap_cons,hi,↓reduceIte]
    · rw [ih]
      rfl
    · exact ih s

def code (h : Host.Call) : List Stmt := [
  .assign .nat [] "pid" (.programId 0),
  .assign .nat [] "batch" (.programId 1),
  .assign .nat [] "chunk_head" (.programId 2),
  .assign .nat [] "tiles_n" (.div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat h.shape.chunk) (.constNat 64)) (.constNat 1)) (.constNat 64)),
  .assign .nat [32] "row" (.add .nat .scalarL
    (.mul .nat .nil (.floorDiv .nat .nil (.ref .nat [] "pid") (.ref .nat [] "tiles_n")) (.constNat 32))
    (.arange 32)),
  .assign .nat [64] "column" (.add .nat .scalarL
    (.mul .nat .nil (.mod .nat .nil (.ref .nat [] "pid") (.ref .nat [] "tiles_n")) (.constNat 64))
    (.arange 64)),
  .assign .nat [32,64] "offsets" (.add .nat Broadcast.nil.consL.consR
    (.add .nat .scalarL
      (.mul .nat .nil (.mul .nat .nil
        (.add .nat .nil (.mul .nat .nil (.ref .nat [] "batch") (.constNat (h.shape.chunks*h.shape.groups)))
          (.ref .nat [] "chunk_head")) (.constNat h.shape.chunk)) (.constNat h.shape.chunk))
      (.mul .nat .scalarR (.expandDim ⟨1,by decide⟩ (.ref .nat [32] "row")) (.constNat h.shape.chunk)))
    (.expandDim ⟨0,by decide⟩ (.ref .nat [64] "column"))),
  .store .real [32,64] (.region (Region.cast h.Out) (.ref .nat [32,64] "offsets"))
    ((Op.const 0.0).broadcast [32,64]) (.mask
      (.boolAnd Broadcast.nil.consSame.consSame
        (.boolAnd Broadcast.nil.consL.consR
          (.lt .nat .scalarR (.expandDim ⟨1,by decide⟩ (.ref .nat [32] "row")) (.constNat h.shape.chunk))
          (.lt .nat .scalarR (.expandDim ⟨0,by decide⟩ (.ref .nat [64] "column")) (.constNat h.shape.chunk)))
        (.gt .nat Broadcast.nil.consR.consL
          (.expandDim ⟨0,by decide⟩ (.ref .nat [64] "column"))
          (.expandDim ⟨1,by decide⟩ (.ref .nat [32] "row"))))) ]

theorem kernel_body (h : Host.Call) : (kernel h).body=code h := by
  simp only [kernel,Kernels.clearUpper,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    Except.bind,bind,eraseDType_float]
  simp only [VeriTile.Triton.eraseDType]
  rfl

theorem program (h : Host.Call) (p : Cleanup.PID h) (s : BlockState) :
    ∃ t, exec (kernel h) (s.withGridIndex p)=some t ∧
      t.mem=(CleanupSource.addresses h p).foldl (store h.Out) s.mem := by
  simp only [exec,kernel_body]
  simp (config := {maxSteps := 300000}) [code,stepStmts,stepStmt,evalOp.eq_def,bind,Option.bind,
    BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,Host.Call.grid,Cleanup.config,
    Grid.rank,GridIndex.toPids,BlockState.setReg,Tile.ptrAdd,Tile.bop,Tile.cop,
    Tile.scalar,Tile.expandDim,TileShape.dropInsertedIndex,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,ComparableDType.le,BlockState.writeMemTyped_real,-Tile.scalar_eta]

  norm_num [CleanupSource.addresses,Cleanup.active,Cleanup.row,Cleanup.column,
    Cleanup.offset,Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,
    NumericDType.div,NumericDType.sub,Nat.add_mul,Nat.mul_assoc,and_assoc]
  exact fold_memory _ _ _ _ _

/-- A complete ordered cleanup grid over the same Real state as the main
kernel. Each element invokes the actual quoted cleanup's Real projection. -/
def schedule (h : Host.Call) : List (Cleanup.PID h) → BlockState → Option BlockState
  | [],s => some s
  | p::ps,s => do
    let next ← exec (kernel h) (s.withGridIndex p)
    schedule h ps next

theorem successful_schedule (h : Host.Call) (order : List (Cleanup.PID h)) (s : BlockState) :
    ∃ t,schedule h order s=some t ∧
      t.mem=(order.flatMap (CleanupSource.addresses h)).foldl (store h.Out) s.mem := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons p ps ih =>
    obtain ⟨middle,hr,hm⟩ := program h p s
    obtain ⟨t,ht,hm'⟩ := ih middle
    refine ⟨t,by simp [schedule,hr,ht],?_⟩
    rw [hm',hm,List.flatMap_cons,List.foldl_append]

theorem fold_at (out : RegionName) (xs : List Nat)
    (memory : RegionName → Nat → MemCell) (r : RegionName) (off : Nat) :
    (xs.foldl (store out) memory) r off=
      if r=out then xs.foldl (Cleanup.store (MemCell.real 0)) (memory r) off else memory r off := by
  induction xs generalizing memory with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons,ih]
    by_cases hr : r=out
    · simp only [hr,↓reduceIte,List.foldl_cons]
      have update : store out memory x out=Cleanup.store (MemCell.real 0) (memory out) x := by
        funext i
        simp [store,Cleanup.store,Function.update_apply]
      rw [update]
    · simp [hr,store]

/-- Exact complete memory bridge from successful cleanup source execution to
the independently proved upper-triangle store fold. -/
theorem complete (h : Host.Call) (s : BlockState) :
    ∃ t,schedule h Finset.univ.toList s=some t ∧
      ∀ r off,t.mem r off=if r=h.Out then
        Cleanup.run h (MemCell.real 0) (s.mem r) off else s.mem r off := by
  obtain ⟨t,ht,hm⟩ := successful_schedule h Finset.univ.toList s
  refine ⟨t,ht,?_⟩
  intro r off
  rw [hm,CleanupSource.addresses_eq,fold_at]
  rfl

#axiomsClean kernel_body
#axiomsClean program
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmChunkFwd.CleanupReal
