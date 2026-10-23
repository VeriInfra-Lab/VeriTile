import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Natural
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Scatter

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def rotationWrites (c : Config) (ops : Ops α) (pid : Program) (key high : Bool) (m : Memory α) :=
  (List.range c.shape.half).map (fun i=>
    let w:=work c.shape pid key i
    ((w.kind.output,c.output w high),component (pairValues c.shape (fun _=>ops) w m) high))
def vWrites (c : Config) (pid : Program) (m : Memory α) :=
  (List.range c.shape.width).map (fun i=>
    ((Region.vcache,c.shape.cache Bool.false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) i),
      m .v (c.shape.data .v pid.2 (pid.1/c.shape.groups) i)))
def naturalProgram (c : Config) (ops : Ops α) (pid : Program) (m : Memory α) : Memory α :=
  let q0:=Stores.commit (rotationWrites c ops pid Bool.false Bool.false m) m
  let q1:=Stores.commit (rotationWrites c ops pid Bool.false Bool.true m) q0
  if pid.1%c.shape.groups=0 then
    let k0:=Stores.commit (rotationWrites c ops pid Bool.true Bool.false m) q1
    let k1:=Stores.commit (rotationWrites c ops pid Bool.true Bool.true m) k0
    Stores.commit (vWrites c pid m) k1
  else q1

theorem rotation_scatter (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (pid : Program) (ht : pid.2<c.shape.tokens) (entry m : Memory α) (key high : Bool)
    (st : State α) (hm : st.mem=lift c m) :
    (scatter (TileShape.allIndices [(parameters c.shape).BLOCK_HALF])
      (if key then "k_cache" else "q") (active (parameters c.shape))
      (if key then cacheOffset (parameters c.shape) (state pid (lift c entry)) high
       else dataOffset (parameters c.shape) (state pid (lift c entry)) Bool.false high)
      (pair rest ops (parameters c.shape) (state pid (lift c entry)) key high) st).mem=
    lift c (Stores.commit (rotationWrites c ops pid key high entry) m) := by
  have h:=scatter_lift c m (if key then .k else .q) c.shape.halfBlock c.shape.half (half_covered c.shape)
    (active (parameters c.shape))
    (if key then cacheOffset (parameters c.shape) (state pid (lift c entry)) high
      else dataOffset (parameters c.shape) (state pid (lift c entry)) Bool.false high)
    (pair rest ops (parameters c.shape) (state pid (lift c entry)) key high)
    (fun i=>c.output (work c.shape pid key i) high)
    (fun i=>component (pairValues c.shape (fun _=>ops) (work c.shape pid key i) entry) high)
    (active_nat c.shape)
    (by intro i hi;have ha: i.1.val<c.shape.half := (active_nat _ _).mp hi
        cases key <;> simp [cacheOffset_nat c hc pid ht entry high i ha,Config.output,work])
    (by intro i hi;exact pair_nat rest ops c pid entry key high i ((active_nat _ _).mp hi)) st hm
  cases key <;> simpa [rotationWrites,work,region,Kind.output,parameters] using h

theorem v_scatter (c : Config) (hc : c.Valid) (pid : Program) (ht : pid.2<c.shape.tokens)
    (entry m : Memory α) (st : State α) (hm : st.mem=lift c m) :
    (scatter (TileShape.allIndices [(parameters c.shape).BLOCK_DIM]) "v_cache"
      (fun i=>i.1.val<(parameters c.shape).HEAD_DIM)
      (valueCacheOffset (parameters c.shape) (state pid (lift c entry)))
      (fun i=>.mk .real (((state pid (lift c entry)).mem "v"
        (valueOffset (parameters c.shape) (state pid (lift c entry)) i)).read .real)) st).mem=
      lift c (Stores.commit (vWrites c pid entry) m) := by
  apply scatter_lift c m .v c.shape.fullBlock c.shape.width (full_covered c.shape)
    _ _ _ (fun i=>c.shape.cache Bool.false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) i)
    (fun i=>entry .v (c.shape.data .v pid.2 (pid.1/c.shape.groups) i))
  · intro i;rfl
  · intro i _;exact valueCacheOffset_nat c hc pid ht entry i
  · intro i _
    rw [valueOffset_nat]
    simp [state,lift,decoded,Cell.read]
  · exact hm

theorem written_memory (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (pid : Program) (ht : pid.2<c.shape.tokens) (m : Memory α) :
    (written rest ops (parameters c.shape) (state pid (lift c m))).mem=
      lift c (naturalProgram c ops pid m) := by
  let s:=state pid (lift c m)
  let q0:=scatter (TileShape.allIndices [c.shape.halfBlock]) "q" (active (parameters c.shape))
    (dataOffset (parameters c.shape) s Bool.false Bool.false)
    (pair rest ops (parameters c.shape) s Bool.false Bool.false) s
  let mq0:=Stores.commit (rotationWrites c ops pid Bool.false Bool.false m) m
  have hq0 : q0.mem=lift c mq0 := rotation_scatter rest ops c hc pid ht m m Bool.false Bool.false s rfl
  let q1:=scatter (TileShape.allIndices [c.shape.halfBlock]) "q" (active (parameters c.shape))
    (dataOffset (parameters c.shape) s Bool.false Bool.true)
    (pair rest ops (parameters c.shape) s Bool.false Bool.true) q0
  let mq1:=Stores.commit (rotationWrites c ops pid Bool.false Bool.true m) mq0
  have hq1 : q1.mem=lift c mq1 := rotation_scatter rest ops c hc pid ht m mq0 Bool.false Bool.true q0 hq0
  by_cases leader : pid.1%c.shape.groups=0
  · let k0:=scatter (TileShape.allIndices [c.shape.halfBlock]) "k_cache" (active (parameters c.shape))
      (cacheOffset (parameters c.shape) s Bool.false)
      (pair rest ops (parameters c.shape) s Bool.true Bool.false) q1
    let mk0:=Stores.commit (rotationWrites c ops pid Bool.true Bool.false m) mq1
    have hk0 : k0.mem=lift c mk0 := rotation_scatter rest ops c hc pid ht m mq1 Bool.true Bool.false q1 hq1
    let k1:=scatter (TileShape.allIndices [c.shape.halfBlock]) "k_cache" (active (parameters c.shape))
      (cacheOffset (parameters c.shape) s Bool.true)
      (pair rest ops (parameters c.shape) s Bool.true Bool.true) k0
    let mk1:=Stores.commit (rotationWrites c ops pid Bool.true Bool.true m) mk0
    have hk1 : k1.mem=lift c mk1 := rotation_scatter rest ops c hc pid ht m mk0 Bool.true Bool.true k0 hk0
    simpa only [written,naturalProgram,state,parameters,leader,ite_true] using
      v_scatter c hc pid ht m mk1 k1 hk1
  · simpa only [written,naturalProgram,state,parameters,leader,ite_true,ite_false] using hq1

theorem vector_readonly (c : Config) (ops : Kind → Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (hr : r≠.q ∧ r≠.kcache ∧ r≠.vcache) :
    vectorBranch c ops ws m r a=m r a := by
  apply vector_frame
  intro w _
  cases kw : w.kind <;> simp_all [Writes,Kind.output]

theorem rotation_branch (c : Config) (hc : c.Valid) (ops : Ops α) (pid : Program)
    (hp : pid∈programs c.shape) (key : Bool) (m : Memory α)
    (guard : key=Bool.false ∨ pid.1%c.shape.groups=0) :
    vectorBranch c (fun _=>ops) (branchWork c.shape pid (if key then .k else .q)) m=
      Stores.commit (rotationWrites c ops pid key Bool.true m)
        (Stores.commit (rotationWrites c ops pid key Bool.false m) m) := by
  rw [branchWork_explicit c.shape hc.shape pid hp]
  cases key <;> simp_all [vectorBranch,vectorWrites,rotationWrites,work,programHead,
    List.map_map,Function.comp_def,Stores.commit,List.foldl_append]

theorem rotation_key_reuse (c : Config) (ops : Ops α) (pid : Program) (high : Bool)
    (m : Memory α) (ws : List Work) :
    rotationWrites c ops pid Bool.true high (vectorBranch c (fun _=>ops) ws m)=
      rotationWrites c ops pid Bool.true high m := by
  simp only [rotationWrites,work,pairValues,Kind.input]
  simp [vector_readonly,show Region.k≠.q ∧ Region.k≠.kcache ∧ Region.k≠.vcache by decide,
    show Region.cos≠.q ∧ Region.cos≠.kcache ∧ Region.cos≠.vcache by decide,
    show Region.sin≠.q ∧ Region.sin≠.kcache ∧ Region.sin≠.vcache by decide]

theorem v_reuse (c : Config) (ops : Ops α) (pid : Program) (m : Memory α) (ws : List Work) :
    vWrites c pid (vectorBranch c (fun _=>ops) ws m)=vWrites c pid m := by
  simp [vWrites,vector_readonly,show Region.v≠.q ∧ Region.v≠.kcache ∧ Region.v≠.vcache by decide]

theorem actualProgram_natural (c : Config) (hc : c.Valid) (ops : Ops α)
    (pid : Program) (hp : pid∈programs c.shape) (m : Memory α) :
    actualProgram c (fun _=>ops) pid m=naturalProgram c ops pid m := by
  have q:=rotation_branch c hc ops pid hp Bool.false m (Or.inl rfl)
  simp only [Bool.false_eq_true,ite_false] at q
  by_cases leader : pid.1%c.shape.groups=0
  · have k:=rotation_branch c hc ops pid hp Bool.true
      (vectorBranch c (fun _=>ops) (branchWork c.shape pid .q) m) (Or.inr leader)
    simp only [rotation_key_reuse,ite_true] at k
    simp only [actualProgram,fullVWrites,leader,ite_true,masked_range _ _ (full_covered c.shape)]
    change Stores.commit (vWrites c pid _) _=_
    rw [v_reuse,v_reuse,k,q]
    simp only [naturalProgram,leader,ite_true]
  · have k : branchWork c.shape pid .k=[] := by
      simp [branchWork_explicit c.shape hc.shape pid hp .k,leader]
    simp [actualProgram,fullVWrites,leader,k,vectorBranch,vectorWrites,Stores.commit,naturalProgram] at q ⊢
    exact q

theorem program_memory (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (pid : Program) (hp : pid∈programs c.shape) (m : Memory α) :
    (execution rest ops (parameters c.shape) (state pid (lift c m))).val.mem=
      lift c (actualProgram c (fun _=>ops) pid m) := by
  rw [execution_memory,written_memory rest ops c hc pid (List.mem_range.mp (List.mem_product.mp hp).2),
    actualProgram_natural c hc ops pid hp]

#axiomsClean program_memory
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
