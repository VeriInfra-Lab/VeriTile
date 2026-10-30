import bench.tritonbench_g.bmm_chunk_fwd.Cleanup
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.BmmChunkFwd.CleanupSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
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
      (Kernels.clearUpper h.Out h.shape.chunk (h.shape.chunks*h.shape.groups)) (atIndex h s p)=some t ∧
      t.mem=(addresses h p).foldl (store h.Out zero) s.mem := by
  simp (config := {maxSteps := 300000}) [FP.Structural.exec,Kernels.clearUpper,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    State.setReg,integralFloorDiv,integralMod,atIndex,algebra,FP.Structural.store,
    GridIndex.toPids,Host.Call.grid,Cleanup.config,Grid.rank]
  have ceiling : h.shape.chunk+64-1=h.shape.chunk+63 := by omega
  simp only [addresses,Cleanup.active,Cleanup.row,Cleanup.column,
    Cleanup.offset,Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,
    Nat.mul_add,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,and_assoc,ceiling]
  exact fold_memory _ _ _ _ _ _

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
      (Kernels.clearUpper h.Out h.shape.chunk (h.shape.chunks*h.shape.groups)) (atIndex h s p)
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
