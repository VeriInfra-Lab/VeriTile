import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPHost
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPReturned
open VeriTile Triton FP.Structural
open FPStorage (Memory state exportMemory)
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def atProgram (c : OutputGrid.Shape) (m : Memory α) (idx : GridIndex (OutputGrid.grid c)) :=
  state m idx.toPids (OutputGrid.grid c).toNumPids

theorem active_eq (c : OutputGrid.Shape) (m : Memory α) (idx : GridIndex (OutputGrid.grid c)) (i : FPOutput.Index c) :
    FPOutput.active c (atProgram c m idx) i=
      RealOutputIO.active (OutputGrid.layout c) (OutputGrid.launchIndex c idx) i := by
  simp [FPOutput.active,FPOutput.row,FPOutput.column,atProgram,state,RealOutputIO.active,
    OutputGrid.layout,OutputGrid.launchIndex,GridIndex.toPids,OutputGrid.grid,Grid.rank]
theorem address_eq (c : OutputGrid.Shape) (m : Memory α) (idx : GridIndex (OutputGrid.grid c)) (i : FPOutput.Index c) :
    FPOutput.outputAddress c (atProgram c m idx) i=
      RealOutputIO.outputAddress (OutputGrid.layout c) (OutputGrid.launchIndex c idx) i := by
  simp [FPOutput.outputAddress,FPOutput.row,FPOutput.column,atProgram,state,RealOutputIO.outputAddress,
    OutputGrid.layout,OutputGrid.launchIndex,GridIndex.toPids,OutputGrid.grid,Grid.rank,Nat.mul_assoc]

theorem address_injective (c : OutputGrid.Shape) (m n : Memory α) (x y : GridIndex (OutputGrid.grid c))
    (i j : FPOutput.Index c) (hi : FPOutput.active c (atProgram c m x) i)
    (hj : FPOutput.active c (atProgram c n y) j)
    (he : FPOutput.outputAddress c (atProgram c m x) i=FPOutput.outputAddress c (atProgram c n y) j) : x=y ∧ i=j := by
  rw [active_eq] at hi hj
  rw [address_eq,address_eq] at he
  exact OutputGrid.address_injective c x y i j hi hj he

def value (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (m : Memory α) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : α :=
  FPOutput.value M D c scale (atProgram c m (OutputGrid.owner c h t v)) (OutputGrid.lane c t v)

theorem value_stable (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (m n : Memory α) (idx : GridIndex (OutputGrid.grid c)) (i : FPOutput.Index c)
    (hm : ∀ r : RegionName, r≠"o" → ∀ off, m r off=n r off) :
    FPOutput.value M D c scale (atProgram c m idx) i=FPOutput.value M D c scale (atProgram c n idx) i := by
  have hq : m "q"=n "q" := funext (hm "q" (by decide))
  have hg : m "g"=n "g" := funext (hm "g" (by decide))
  have hh : m "h"=n "h" := funext (hm "h" (by decide))
  have hv : m "v"=n "v" := funext (hm "v" (by decide))
  have ha : m "attention"=n "attention" := funext (hm "attention" (by decide))
  have hquery (f : Nat) : FPOutput.queryGate M c scale (atProgram c m idx) f=
      FPOutput.queryGate M c scale (atProgram c n idx) f := by
    funext j
    simp [FPOutput.queryGate,FPOutput.featureLoad,FPOutput.row,atProgram,state,FPStorage.importMemory,hq,hg]
  have hhistory (f : Nat) : FPOutput.historyLoad M c (atProgram c m idx) f=
      FPOutput.historyLoad M c (atProgram c n idx) f := by
    funext j
    simp [FPOutput.historyLoad,FPOutput.column,atProgram,state,FPStorage.importMemory,hh]
  have hupdate : FPOutput.update M D c scale (atProgram c m idx)=FPOutput.update M D c scale (atProgram c n idx) := by
    funext f a j
    simp only [FPOutput.update,hquery,hhistory]
  have hattention : FPOutput.attentionLoad M c (atProgram c m idx)=FPOutput.attentionLoad M c (atProgram c n idx) := by
    funext j
    simp [FPOutput.attentionLoad,FPOutput.row,atProgram,state,FPStorage.importMemory,ha]
  have hvalue : FPOutput.valueLoad M c (atProgram c m idx)=FPOutput.valueLoad M c (atProgram c n idx) := by
    funext j
    simp [FPOutput.valueLoad,FPOutput.active,FPOutput.outputAddress,FPOutput.row,FPOutput.column,
      atProgram,state,FPStorage.importMemory,hv]
  simp only [FPOutput.value,FPOutput.accumulated,hupdate,hattention,hvalue]

theorem grid_readback (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (m : Memory α) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    FPStages.reference M D (.output c scale) m "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)=value M D c scale m h t v := by
  classical
  let owner := OutputGrid.owner c h t v
  let lane := OutputGrid.lane c t v
  let off := h.val*(c.T*c.V)+t.val*c.V+v.val
  have ha (n : Memory α) : FPOutput.active c (atProgram c n owner) lane := by
    rw [active_eq]
    exact OutputGrid.owner_active c h t v
  have ho (n : Memory α) : FPOutput.outputAddress c (atProgram c n owner) lane=off := by
    rw [address_eq]
    exact OutputGrid.owner_address c h t v
  let I := fun n : Memory α => ∀ r : RegionName, r≠"o" → ∀ o, n r o=m r o
  apply FPGrid.reference_readback (FPStages.referenceStep M D (.output c scale))
    (FPGrid.jobs (OutputGrid.grid c)) owner ((FPGrid.jobs_complete _).2 owner)
    (FPGrid.jobs_complete _).1 m I (by intro r hr o; rfl) "o" off
  · intro idx _ n hn r hr o
    exact (FPStages.referenceStep_frame M D (.output c scale) idx n r o hr).trans (hn r hr o)
  · intro n hn
    change FPStorage.payload ((FPOutput.written M D c scale (atProgram c n owner)).mem "o" off)=_
    rw [← ho n,FPOutput.readback M D c scale (atProgram c n owner) lane (ha n) (by
      intro i j hi hj he
      exact (address_injective c n n owner owner i j hi hj he).2)]
    exact value_stable M D c scale n m owner lane hn
  · intro idx _ hne n
    change FPStorage.payload ((FPOutput.written M D c scale (atProgram c n idx)).mem "o" off)=n "o" off
    rw [FPOutput.frame M D c scale (atProgram c n idx) "o" off (Or.inr (by
      intro j hj he
      have he' : FPOutput.outputAddress c (atProgram c n owner) lane=FPOutput.outputAddress c (atProgram c n idx) j :=
        (ho n).trans he
      exact hne (address_injective c n n owner idx lane j (ha n) hj he').1.symm))]
    simp [atProgram,state,FPStorage.importMemory]

theorem consumer_readback (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions) (scale : ℝ)
    (m : Memory α) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    Host.consumerReference M D c scale m "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)=
      value M D c.consumer scale m h t v := by
  have hh : 0<c.heads := Nat.zero_lt_of_lt h.isLt
  have ht : 0<c.T := Nat.zero_lt_of_lt t.isLt
  have hv : 0<c.V := Nat.zero_lt_of_lt v.isLt
  have hp : 0<Host.outputSize c := Nat.mul_pos (Nat.mul_pos hh ht) hv
  simp only [Host.consumerReference,Nat.ne_of_gt hp,if_false]
  exact grid_readback M D c.consumer scale m h t v

theorem composed_readback (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions) (scale : ℝ)
    (m : Memory α) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    Host.consumerReference M D c scale (Host.producerReference M D c scale m) "o"
      (h.val*(c.T*c.V)+t.val*c.V+v.val)=
      value M D c.consumer scale (Host.producerReference M D c scale m) h t v :=
  consumer_readback M D c scale _ h t v

#specNonCircular value avoiding [FPStages.execute,FPGrid.invoke,FP.Structural.exec]
#axiomsClean grid_readback
#axiomsClean composed_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPReturned
