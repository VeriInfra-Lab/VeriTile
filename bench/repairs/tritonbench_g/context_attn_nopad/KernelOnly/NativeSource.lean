import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Safety
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Native

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.NativeSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000

/-- The repaired source's three explicit signed64 PID casts. -/
def pids (r : Request) (idx : GridIndex r.call.grid) : Nat → Nat
  | 0 => (Safety.word (r.call.launch idx).batch).toInt.toNat
  | 1 => (Safety.word (r.call.launch idx).head).toInt.toNat
  | 2 => (Safety.word (r.call.launch idx).block).toInt.toNat
  | _ => 0

theorem pids_eq (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid) :
    pids r idx=idx.toPids := by
  have projected := Safety.projects r heap h idx
  funext axisId
  rcases axisId with _ | (_ | (_ | axisId))
  · simp only [pids,projected.batch,Int.toNat_natCast]; rfl
  · simp only [pids,projected.head,Int.toNat_natCast]; rfl
  · simp only [pids,projected.block,Int.toNat_natCast]; rfl
  · simp [pids,GridIndex.toPids,Call.grid,Grid.rank]

def atProgram (r : Request) (idx : GridIndex r.call.grid) (s : State NativePhysical.Expr) : State NativePhysical.Expr :=
  {s with pids := pids r idx,numPids := r.call.grid.toNumPids}

/-- Actual source execution after evaluating the finite PID prefix. Remaining
finite control/metadata/formed pointer sites are exactly Safety.Projects. -/
def runGrid (r : Request) : List (GridIndex r.call.grid) → State NativePhysical.Expr → Option (State NativePhysical.Expr)
  | [],s => some s
  | idx::rest,s => do
    let t ← FP.Structural.exec (FP.ProfiledRewrite.engine (backend r).algebra)
      (kernel r.call regions) (atProgram r idx s)
    runGrid r rest t

theorem runGrid_eq (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (jobs : List (GridIndex r.call.grid)) (s : State NativePhysical.Expr) :
    runGrid r jobs s=FPReference.runGrid r.call regions (backend r) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    have prepared : atProgram r idx s=FPHost.atProgram r.call idx s := by
      simp only [atProgram,FPHost.atProgram,pids_eq r heap h idx]
    simp only [runGrid,FPReference.runGrid,prepared]
    cases FP.Structural.exec (FP.ProfiledRewrite.engine (backend r).algebra)
        (kernel r.call regions) (FPHost.atProgram r.call idx s) with
    | none => rfl
    | some t => exact ih t

theorem source_connection (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    runGrid r (order r) (initial r heap)=execute r heap ∧
      ∀ idx : GridIndex r.call.grid,Safety.Projects r heap idx :=
  ⟨runGrid_eq r heap h _ _,Safety.projects r heap h⟩

#axiomsClean source_connection
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.NativeSource
