import bench.tritonbench_g.bmm_optimized.Proofs.Reference
import bench.tritonbench_g.bmm_optimized.Sources.SourceIndex

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
open VeriTile.Triton FP.Structural
open BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- Actual submitted PID and dimensions; register reset is per CTA. -/
def start (s : Shape) (c : Config) (pid : Program) (state : State α) : State α :=
  {state with
    regs:=fun _ _ _=>none
    pids:=fun ax=>if ax=0 then pid.1 else if ax=1 then pid.2.1 else if ax=2 then pid.2.2 else 0
    numPids:=fun ax=>if ax=0 then s.gx c else if ax=1 then s.gy c else if ax=2 then s.batch else 1}

def laneOf {c : Config} (i : TileIndex [c.tm,c.tn]) : Lane := (i.1.val,i.2.1.val)
theorem lane_mem (c : Config) (i : TileIndex [c.tm,c.tn]) : laneOf i∈lanes c := by
  simp [laneOf,lanes,i.1.isLt,i.2.1.isLt]

structure Related (m : Memory α) (state : State α) : Prop where
  a : ∀ off,state.mem "A" off=.mk .real (m .a off)
  b : ∀ off,state.mem "B" off=.mk .real (m .b off)
  output : ∀ off,state.mem "O" off=.mk .real (m .output off)

private theorem optionalMask (n tile coordinate : Nat) :
    (decide (n%tile=0)=Bool.true ∨ coordinate<n) ↔ axisMask n tile coordinate := by
  unfold axisMask
  split_ifs <;> simp_all

theorem row_eq (s : Shape) (c : Config) (pid : Program) (state : State α)
    (i : TileIndex [c.tm,c.tn]) :
    row (parameters s c) (start s c pid state) i.1=(cellOf s c pid (laneOf i)).row := rfl
theorem column_eq (s : Shape) (c : Config) (pid : Program) (state : State α)
    (i : TileIndex [c.tm,c.tn]) :
    column (parameters s c) (start s c pid state) i.2.1=(cellOf s c pid (laneOf i)).column := rfl

theorem a_offset (s : Shape) (c : Config) (pid : Program) (state : State α)
    (i : TileIndex [c.tm,c.tn]) (k : Nat) (j : Fin 32) :
    aOffset (parameters s c) (start s c pid state) k (i.1,j,PUnit.unit)=
      s.a pid.2.2 (cellOf s c pid (laneOf i)).row (k*32+j.val) := by
  change pid.2.2*s.m*s.k+(cellOf s c pid (laneOf i)).row*s.k+j.val+k*32=_
  unfold Shape.a
  ring
theorem b_offset (s : Shape) (c : Config) (pid : Program) (state : State α)
    (i : TileIndex [c.tm,c.tn]) (k : Nat) (j : Fin 32) :
    bOffset (parameters s c) (start s c pid state) k (j,i.2.1,PUnit.unit)=
      s.b pid.2.2 (k*32+j.val) (cellOf s c pid (laneOf i)).column := by
  change pid.2.2*s.k*s.n+j.val*s.n+(cellOf s c pid (laneOf i)).column+k*(32*s.n)=_
  unfold Shape.b
  ring

theorem source_loads (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (hp : pid∈programs s c) (m : Memory α) (state : State α) (related : Related m state)
    (i : TileIndex [c.tm,c.tn]) (k : Nat) (hk : k<s.iterations) (j : Fin 32) :
    aValue o (parameters s c) (start s c pid state) k (i.1,j,PUnit.unit)=
      aLoad s o m (cellOf s c pid (laneOf i)) (k*32+j.val) ∧
    bValue o (parameters s c) (start s c pid state) k (j,i.2.1,PUnit.unit)=
      bLoad s o m (cellOf s c pid (laneOf i)) (k*32+j.val) := by
  have masks:=source_masks s c pid hp (laneOf i) (lane_mem c i) k j.val hk j.isLt
  have roweq:=row_eq s c pid state i
  have coleq:=column_eq s c pid state i
  constructor
  · unfold aValue
    have test : ((parameters s c).divM=Bool.true ∨ row (parameters s c) (start s c pid state) i.1<(parameters s c).M) ∧
        ((parameters s c).divK=Bool.true ∨ inner (parameters s c) k j<(parameters s c).K) ↔
        (cellOf s c pid (laneOf i)).row<s.m ∧ k*32+j.val<s.k := by
      change ((decide (s.m%c.tm=0)=Bool.true ∨ (cellOf s c pid (laneOf i)).row<s.m) ∧
        (decide (s.k%32=0)=Bool.true ∨ k*32+j.val<s.k)) ↔ _
      rw [optionalMask,optionalMask]
      exact masks.1
    simp only [test,a_offset s c pid state i k j]
    simp only [start,parameters,related.a,Cell.read_same,aLoad,cellOf]
  · unfold bValue
    have test : ((parameters s c).divK=Bool.true ∨ inner (parameters s c) k j<(parameters s c).K) ∧
        ((parameters s c).divN=Bool.true ∨ column (parameters s c) (start s c pid state) i.2.1<(parameters s c).N) ↔
        k*32+j.val<s.k ∧ (cellOf s c pid (laneOf i)).column<s.n := by
      change ((decide (s.k%32=0)=Bool.true ∨ k*32+j.val<s.k) ∧
        (decide (s.n%c.tn=0)=Bool.true ∨ (cellOf s c pid (laneOf i)).column<s.n)) ↔ _
      rw [optionalMask,optionalMask]
      exact masks.2.1
    simp only [test,b_offset s c pid state i k j]
    simp only [start,parameters,related.b,Cell.read_same,bLoad,cellOf]

#axiomsClean source_loads
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
