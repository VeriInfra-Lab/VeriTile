import bench.tritonbench_g.bmm_chunk_fwd.Proofs.CleanupSource
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Cleanup
namespace VeriTile.Bench.Repairs.BmmChunkFwd.CleanupReal
open VeriTile.Triton
open ControlArithmetic
attribute [local simp 2000] VeriTile.Triton.eval_truncDiv VeriTile.Triton.eval_truncMod
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 300000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def kernel (h : Host.Call) : Kernel :=
  (Kernels.clearUpperKernel h.Out h.shape.chunk (h.shape.chunks*h.shape.groups)).eraseDType.toAlgKernel

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

/-- Only the Real projection of the repaired cleanup source is retained. -/
def code (h : Host.Call) : List Stmt := (kernel h).body

theorem kernel_body (h : Host.Call) : (kernel h).body=code h := rfl

theorem program (h : Host.Call) (p : Cleanup.PID h) (s : BlockState) :
    ∃ t, exec (kernel h) (s.withGridIndex p)=some t ∧
      t.mem=(CleanupSource.addresses h p).foldl (store h.Out) s.mem := by
  simp (config := {maxSteps := 300000}) [exec,kernel_body,code,kernel,Kernels.clearUpperKernel,
    VeriTile.Bench.Repairs.BmmChunkFwd.clearUpperKernel,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType,Except.bind,eraseDType_float,
    stepStmts,stepStmt,evalOp.eq_def,bind,Option.bind,
    BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,Host.Call.grid,Cleanup.config,
    Grid.rank,GridIndex.toPids,Tile.ptrAdd,Tile.bop,Tile.cop,Tile.uop,Tile.select,
    Tile.scalar,Tile.expandDim,TileShape.dropInsertedIndex,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
    ComparableDType.lt,ComparableDType.le,BlockState.writeMemTyped_real,-Tile.scalar_eta]
  simp only [show (0.0 : ℝ)=0 by norm_num]
  have ceiling : h.shape.chunk+64-1=h.shape.chunk+63 := by omega
  simp only [CleanupSource.addresses,Cleanup.active,Cleanup.row,Cleanup.column,
    Cleanup.offset,Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,
    Nat.mul_add,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,and_assoc,ceiling]
  have hcdiv : Int.tdiv ((h.shape.chunk : Int)+63) 64 =
      (Host.cdiv h.shape.chunk 64 : Int) := by
    simpa only [Host.cdiv,ceiling,show (64 : Int)-1=63 by decide] using
      (ControlArithmetic.cdiv_signed_int h.shape.chunk 64)
  have row_cast (i : Fin 32) :
      (Cleanup.pid0 p : Int)/(Host.cdiv h.shape.chunk 64 : Int)*32+(i.val : Int) =
        (Cleanup.row h p i : Int) := by
    simp [Cleanup.row,Cleanup.pid0]
  have col_cast (j : Fin 64) :
      (Cleanup.pid0 p : Int).tmod (Host.cdiv h.shape.chunk 64 : Int)*64+(j.val : Int) =
        (Cleanup.column h p j : Int) := by
    simp [Cleanup.column,Cleanup.pid0,Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg _)]
  have offset_cast (i : Fin 32) (j : Fin 64) :
      Int.toNat (((Cleanup.pid1 p : Int)*(h.shape.chunks*h.shape.groups : Int)+(Cleanup.pid2 p : Int))*
        (h.shape.chunk : Int)*(h.shape.chunk : Int)+(Cleanup.row h p i : Int)*(h.shape.chunk : Int)+
        (Cleanup.column h p j : Int))=Cleanup.offset h p i j := by
    have eq : ((Cleanup.pid1 p : Int)*(h.shape.chunks*h.shape.groups : Int)+(Cleanup.pid2 p : Int))*
        (h.shape.chunk : Int)*(h.shape.chunk : Int)+(Cleanup.row h p i : Int)*(h.shape.chunk : Int)+
        (Cleanup.column h p j : Int)=(Cleanup.offset h p i j : Int) := by
      simp [Cleanup.offset,Cleanup.pid1,Cleanup.pid2,Nat.add_mul]
      ring
    rw [eq];rfl
  have mask_bridge (i : Fin 32) (j : Fin 64) :
      ((Cleanup.pid0 p : Int)/(Host.cdiv h.shape.chunk 64 : Int)*32+(i.val : Int)<(h.shape.chunk : Int) ∧
       (Cleanup.pid0 p : Int).tmod (Host.cdiv h.shape.chunk 64 : Int)*64+(j.val : Int)<(h.shape.chunk : Int) ∧
       (Cleanup.pid0 p : Int)/(Host.cdiv h.shape.chunk 64 : Int)*32+(i.val : Int)<
         (Cleanup.pid0 p : Int).tmod (Host.cdiv h.shape.chunk 64 : Int)*64+(j.val : Int)) ↔
        Cleanup.active h p i j := by
    rw [row_cast i,col_cast j]
    simp only [Int.ofNat_lt,Cleanup.active]
  have address_bridge (i : Fin 32) (j : Fin 64) :
      Int.toNat (((Cleanup.pid1 p : Int)*(h.shape.chunks*h.shape.groups : Int)+(Cleanup.pid2 p : Int))*
        (h.shape.chunk : Int)*(h.shape.chunk : Int)+
        ((Cleanup.pid0 p : Int)/(Host.cdiv h.shape.chunk 64 : Int)*32+(i.val : Int))*(h.shape.chunk : Int)+
        ((Cleanup.pid0 p : Int).tmod (Host.cdiv h.shape.chunk 64 : Int)*64+(j.val : Int)))=
          Cleanup.offset h p i j := by
    rw [row_cast i,col_cast j]
    exact offset_cast i j
  simp only [Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,ceiling,Int.natCast_mul] at mask_bridge address_bridge
  try simp only [hcdiv,Host.cdiv,ceiling,mask_bridge,address_bridge]
  let tiles : Nat := (h.shape.chunk+63)/64
  let rr (i : Fin 32) : Int := (Cleanup.pid0 p : Int)/(tiles : Int)*32+(i.val : Int)
  let cc (j : Fin 64) : Int := (Cleanup.pid0 p : Int).tmod (tiles : Int)*64+(j.val : Int)
  let rawMask (i : TileIndex [32,64]) : Prop :=
    rr i.1<(h.shape.chunk : Int) ∧ cc i.2.1<(h.shape.chunk : Int) ∧ rr i.1<cc i.2.1
  let rawOffset (i : TileIndex [32,64]) : Nat :=
    Int.toNat (((Cleanup.pid1 p : Int)*(h.shape.chunks*h.shape.groups : Int)+(Cleanup.pid2 p : Int))*
      (h.shape.chunk : Int)*(h.shape.chunk : Int)+rr i.1*(h.shape.chunk : Int)+cc i.2.1)
  have hmask (i : TileIndex [32,64]) : rawMask i ↔ Cleanup.active h p i.1 i.2.1 := by
    simpa only [rawMask,rr,cc,tiles,Host.cdiv,ceiling] using mask_bridge i.1 i.2.1
  have hoff (i : TileIndex [32,64]) : rawOffset i=Cleanup.offset h p i.1 i.2.1 := by
    simpa only [rawOffset,rr,cc,tiles,Host.cdiv,ceiling] using address_bridge i.1 i.2.1
  have bridge (jobs : List (TileIndex [32,64])) (entry : BlockState) :
      (jobs.foldl (fun t i => if rawMask i then t.writeMem h.Out (rawOffset i) 0 else t) entry).mem=
        (jobs.filterMap (fun i => if Cleanup.active h p i.1 i.2.1 then
          some (Cleanup.offset h p i.1 i.2.1) else none)).foldl (store h.Out) entry.mem := by
    induction jobs generalizing entry with
    | nil => rfl
    | cons i jobs ih =>
      by_cases ha : Cleanup.active h p i.1 i.2.1
      · have hm : rawMask i := (hmask i).mpr ha
        simp only [List.foldl_cons,List.filterMap_cons,ha,hm,↓reduceIte]
        rw [ih,hoff i]
        rfl
      · have hm : ¬rawMask i := fun hn => ha ((hmask i).mp hn)
        simpa only [List.foldl_cons,List.filterMap_cons,ha,hm,↓reduceIte] using ih entry
  simpa only [rawMask,rawOffset,rr,cc,tiles,Cleanup.active,Cleanup.row,Cleanup.column,Cleanup.offset,
    Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,ceiling,BlockState.setReg_mem,
    Nat.mul_add,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,and_assoc,Int.natCast_mul] using
    (bridge (TileShape.allIndices [32,64]) _)

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
