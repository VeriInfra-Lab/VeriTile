import bench.tritonbench_g.bmm_optimized.Proofs.Operands

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
open VeriTile.Triton FP.Structural
open BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem source_fma_fold (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (hp : pid∈programs s c) (m : Memory α) (state : State α) (related : Related m state)
    (i : TileIndex [c.tm,c.tn]) (k : Nat) (hk : k<s.iterations) (n : Nat) (hn : n≤32) (seed : α) :
    fmaFold o
      (fun j=>if h : j<32 then aValue o (parameters s c) (start s c pid state) k (i.1,⟨j,h⟩,PUnit.unit) else o.zero)
      (fun j=>if h : j<32 then bValue o (parameters s c) (start s c pid state) k (⟨j,h⟩,i.2.1,PUnit.unit) else o.zero)
      n seed=laneLoop s o m (cellOf s c pid (laneOf i)) (k*32) n seed := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    have hlt : n<32 := by omega
    have loads:=source_loads s c o pid hp m state related i k hk ⟨n,hlt⟩
    simp only [fmaFold,laneLoop,ih (by omega),dif_pos hlt]
    exact congrArg₂ (fun a b=>o.fma a b (laneLoop s o m (cellOf s c pid (laneOf i)) (k*32) n seed)) loads.1 loads.2

theorem source_accumulation (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (hp : pid∈programs s c) (m : Memory α) (state : State α) (related : Related m state)
    (i : TileIndex [c.tm,c.tn]) (n : Nat) (hn : n≤s.iterations) :
    accumulation o (parameters s c) (start s c pid state) n i=
      blockLoop s o m (cellOf s c pid (laneOf i)) n := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    simp only [accumulation,dotTile,ih (by omega),blockLoop]
    exact source_fma_fold s c o pid hp m state related i n (by omega) 32 (by omega) _

theorem source_value (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (hp : pid∈programs s c) (m : Memory α) (state : State α) (related : Related m state)
    (i : TileIndex [c.tm,c.tn]) :
    accumulation o (parameters s c) (start s c pid state) (iterations (parameters s c)) i=
      sourceValue s o m (cellOf s c pid (laneOf i)) :=
  source_accumulation s c o pid hp m state related i s.iterations (by omega)

theorem active_exact (s : Shape) (c : Config) (pid : Program) (hp : pid∈programs s c)
    (state : State α) (i : TileIndex [c.tm,c.tn]) :
    Execution.active (parameters s c) (start s c pid state) i ↔ s.Active (cellOf s c pid (laneOf i)) := by
  have formed:=source_lane_tiles s c pid hp (laneOf i) (lane_mem c i)
  have cb:=config_bounds c
  have rm:=axisMask_exact s.m c.tm _ cb.1 formed.1
  have cm:=axisMask_exact s.n c.tn _ cb.2.2.1 formed.2.1
  have opt (n tile coordinate : Nat) : (decide (n%tile=0)=Bool.true ∨ coordinate<n) ↔ axisMask n tile coordinate := by
    unfold axisMask
    split_ifs <;> simp_all
  unfold Execution.active
  rw [row_eq,column_eq]
  simp only [parameters,opt,rm,cm,Shape.Active,formed.2.2,true_and]

theorem o_offset (s : Shape) (c : Config) (pid : Program) (state : State α)
    (i : TileIndex [c.tm,c.tn]) :
    oOffset (parameters s c) (start s c pid state) i=
      s.o pid.2.2 (cellOf s c pid (laneOf i)).row (cellOf s c pid (laneOf i)).column := by
  change pid.2.2*s.m*s.n+(cellOf s c pid (laneOf i)).row*s.n+(cellOf s c pid (laneOf i)).column=_
  unfold Shape.o
  ring

#axiomsClean source_value
#axiomsClean active_exact
#axiomsClean o_offset
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
