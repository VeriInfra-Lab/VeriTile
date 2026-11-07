import bench.tritonbench_g.bmm_chunk_fwd.Proofs.ControlArithmetic
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Support.Cleanup
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.BmmChunkFwd.CleanupSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open ControlArithmetic
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- The cleanup has only the FP32 constant +0. The provided zero is its value
in the selected carrier; no arithmetic, load, reduction, or cast result is a port. -/
def algebra {α : Type} (rest : Algebra α) (zero : α) : Algebra α :=
  {rest with literal:=fun _ _ _ => zero,cast:=fun _ _ _ x => x}

def atIndex {α : Type} (h : Host.Call) (s : State α) (p : Cleanup.PID h) : State α :=
  {s with regs:=fun _ _ _ => none,pids:=p.toPids,numPids:=(h.grid Cleanup.config).toNumPids}

def addresses (h : Host.Call) (p : Cleanup.PID h) : List Nat :=
  (TileShape.allIndices [32,64]).filterMap fun i =>
    if Cleanup.active h p i.1 i.2.1 then some (Cleanup.offset h p i.1 i.2.1) else none

def store {α : Type} (out : RegionName) (zero : α)
    (memory : RegionName → Nat → Cell α) (address : Nat) : RegionName → Nat → Cell α :=
  fun r off => if r=out ∧ off=address then .mk .fp32 (ofFloat .fp32 zero) else memory r off

theorem fold_memory {α ι : Type} (out : RegionName) (zero : α) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write out (off i) (.mk .fp32 (ofFloat .fp32 zero)) else t) s).mem=
      (l.filterMap (fun i => if active i then some (off i) else none)).foldl (store out zero) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,List.filterMap_cons,hi,↓reduceIte]
    · rw [ih]
      rfl
    · exact ih s

/-- Successful execution of the actual readable cleanup DSL at an actual grid
coordinate, with the exact source's masked store list and complete memory. -/
theorem program {α : Type} [Inhabited α] (rest : Algebra α) (zero : α)
    (h : Host.Call) (p : Cleanup.PID h) (s : State α) :
    ∃ t, FP.Structural.exec (engine (algebra rest zero))
      (Kernels.clearUpperKernel h.Out h.shape.chunk (h.shape.chunks*h.shape.groups)) (atIndex h s p)=some t ∧
      t.mem=(addresses h p).foldl (store h.Out zero) s.mem := by
  simp (config := {maxSteps := 300000}) [FP.Structural.exec,Kernels.clearUpperKernel,VeriTile.Bench.Repairs.BmmChunkFwd.clearUpperKernel,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    integralFloorDiv,integralMod,atIndex,algebra,FP.Structural.store,
    GridIndex.toPids,Host.Call.grid,Cleanup.config,Grid.rank]
  have ceiling : h.shape.chunk+64-1=h.shape.chunk+63 := by omega
  simp only [addresses,Cleanup.active,Cleanup.row,Cleanup.column,
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
  simp only [hcdiv,Host.cdiv,ceiling,mask_bridge,address_bridge]
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
  have bridge (jobs : List (TileIndex [32,64])) (entry : State α) :
      (jobs.foldl (fun t i => if rawMask i then t.write h.Out (rawOffset i) (.mk .fp32 zero) else t) entry).mem=
        (jobs.filterMap (fun i => if Cleanup.active h p i.1 i.2.1 then
          some (Cleanup.offset h p i.1 i.2.1) else none)).foldl (store h.Out zero) entry.mem := by
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
    Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,ceiling,State.setReg_mem,
    Nat.mul_add,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,and_assoc,Int.natCast_mul] using
    (bridge (TileShape.allIndices [32,64]) _)

/-- The readable DSL's row-major lanes are exactly the saved source event list. -/
theorem addresses_eq (h : Host.Call) :
    (Finset.univ.toList : List (Cleanup.PID h)).flatMap (addresses h)=Cleanup.addresses h := by
  unfold Cleanup.addresses
  apply congrArg (fun f => (Finset.univ.toList : List (Cleanup.PID h)).flatMap f)
  funext p
  simp only [addresses,TileShape.allIndices,List.flatMap_map,
    List.map_flatMap,List.map_map,List.map_cons,List.map_nil,List.flatMap_cons,
    List.flatMap_nil,List.append_nil,List.filterMap_flatMap,List.filterMap_map,
    List.filterMap_cons,List.filterMap_nil]
  congr 1
  funext i
  induction List.finRange 64 with
  | nil => rfl
  | cons j js ih =>
    simp only [List.flatMap_cons,List.filterMap_cons]
    split <;> simp_all

/-- Full cleanup launch. Registers are local to each actual program; memory is
threaded through the exact all-program grid schedule. -/
def schedule {α : Type} [Inhabited α] (rest : Algebra α) (zero : α) (h : Host.Call) :
    List (Cleanup.PID h) → State α → Option (State α)
  | [],s => some s
  | p::ps,s => do
    let next ← FP.Structural.exec (engine (algebra rest zero))
      (Kernels.clearUpperKernel h.Out h.shape.chunk (h.shape.chunks*h.shape.groups)) (atIndex h s p)
    schedule rest zero h ps next

theorem successful_schedule {α : Type} [Inhabited α] (rest : Algebra α) (zero : α)
    (h : Host.Call) (order : List (Cleanup.PID h)) (s : State α) :
    ∃ t,schedule rest zero h order s=some t ∧
      t.mem=(order.flatMap (addresses h)).foldl (store h.Out zero) s.mem := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons p ps ih =>
    obtain ⟨mid,hr,hm⟩ := program rest zero h p s
    obtain ⟨t,ht,hm'⟩ := ih mid
    refine ⟨t,by simp [schedule,hr,ht],?_⟩
    rw [hm',hm,List.flatMap_cons,List.foldl_append]

theorem fold_at {α : Type} (out : RegionName) (zero : α) (xs : List Nat)
    (memory : RegionName → Nat → Cell α) (r : RegionName) (off : Nat) :
    (xs.foldl (store out zero) memory) r off=
      if r=out then xs.foldl (Cleanup.store (.mk .fp32 (ofFloat .fp32 zero))) (memory r) off
      else memory r off := by
  induction xs generalizing memory with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons,ih]
    by_cases hr : r=out
    · simp only [hr,↓reduceIte,List.foldl_cons]
      have update : store out zero memory x out=
          Cleanup.store (.mk .fp32 (ofFloat .fp32 zero)) (memory out) x := by
        funext i
        simp [store,Cleanup.store,Function.update_apply]
      rw [update]
    · simp [hr,store]

theorem complete {α : Type} [Inhabited α] (rest : Algebra α) (zero : α)
    (h : Host.Call) (s : State α) :
    ∃ t,schedule rest zero h Finset.univ.toList s=some t ∧
      ∀ r off,t.mem r off=if r=h.Out then
        Cleanup.run h (.mk .fp32 (ofFloat .fp32 zero)) (s.mem r) off else s.mem r off := by
  obtain ⟨t,ht,hm⟩ := successful_schedule rest zero h Finset.univ.toList s
  refine ⟨t,ht,?_⟩
  intro r off
  rw [hm,addresses_eq,fold_at]
  rfl

#axiomsClean program
#axiomsClean addresses_eq
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmChunkFwd.CleanupSource
