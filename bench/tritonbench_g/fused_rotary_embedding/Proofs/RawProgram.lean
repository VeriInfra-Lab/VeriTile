import bench.tritonbench_g.fused_rotary_embedding.Proofs.RawMemory
import bench.tritonbench_g.fused_rotary_embedding.InvocationIO

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The already-proved five store groups, in source order. All values refer
to this program's entry memory, as established by `execution_memory`. -/
def programWrites (c : Config) (ops : Ops α) (pid : Program) (m : Memory α) :=
  rotationWrites c ops pid Bool.false Bool.false m ++
  rotationWrites c ops pid Bool.false Bool.true m ++
  if pid.1%c.shape.groups=0 then
    rotationWrites c ops pid Bool.true Bool.false m ++
    rotationWrites c ops pid Bool.true Bool.true m ++ vWrites c pid m
  else []

def rawProgram (c : Config) (ops : Ops α) (pid : Program) (m : Mem α) :=
  rawCommit (programWrites c ops pid (rawValues m)) m

theorem programWrites_outputs (c : Config) (ops : Ops α) (pid : Program) (m : Memory α)
    (w : Stores.Write α) (hw : w ∈ programWrites c ops pid m) :
    ∃ kind : Kind, w.1.1 = kind.output := by
  have rot (key high : Bool) (w : Stores.Write α) (hw : w ∈ rotationWrites c ops pid key high m) :
      ∃ kind : Kind, w.1.1 = kind.output := by
    rcases List.mem_map.mp hw with ⟨i,_,rfl⟩
    exact ⟨(work c.shape pid key i).kind,rfl⟩
  have val (w : Stores.Write α) (hw : w ∈ vWrites c pid m) :
      ∃ kind : Kind, w.1.1 = kind.output := by
    rcases List.mem_map.mp hw with ⟨i,_,rfl⟩
    exact ⟨.v,rfl⟩
  simp only [programWrites,List.mem_append] at hw
  rcases hw with (hw|hw)|hw
  · exact rot _ _ _ hw
  · exact rot _ _ _ hw
  · split_ifs at hw with leader
    · rcases List.mem_append.mp hw with hw|hw
      · rcases List.mem_append.mp hw with hw|hw <;> exact rot _ _ _ hw
      · exact val _ hw
    · simp at hw

theorem rawProgram_values (c : Config) (ops : Ops α) (pid : Program) (m : Mem α) :
    rawValues (rawProgram c ops pid m) = naturalProgram c ops pid (rawValues m) := by
  rw [rawProgram,rawCommit_values _ (programWrites_outputs c ops pid (rawValues m))]
  by_cases leader : pid.1%c.shape.groups=0 <;>
    simp [programWrites,naturalProgram,leader,Stores.commit,List.foldl_append]

theorem rawProgram_metadata (c : Config) (ops : Ops α) (pid : Program) (m : Mem α)
    (hm : MetadataBinding c m) : MetadataBinding c (rawProgram c ops pid m) :=
  rawCommit_metadata c _ (programWrites_outputs c ops pid (rawValues m)) m hm

theorem raw_rotation_scatter (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (pid : Program) (ht : pid.2 < c.shape.tokens) (entry : Mem α)
    (hm : MetadataBinding c entry) (key high : Bool) (st : State α) :
    (scatter (TileShape.allIndices [(parameters c.shape).BLOCK_HALF])
      (if key then "k_cache" else "q") (active (parameters c.shape))
      (if key then cacheOffset (parameters c.shape) (state pid entry) high
       else dataOffset (parameters c.shape) (state pid entry) Bool.false high)
      (pair rest ops (parameters c.shape) (state pid entry) key high) st).mem =
    rawCommit (rotationWrites c ops pid key high (rawValues entry)) st.mem := by
  have h := raw_scatter (if key then .k else .q) c.shape.halfBlock c.shape.half (half_covered c.shape)
    (active (parameters c.shape))
    (if key then cacheOffset (parameters c.shape) (state pid entry) high
      else dataOffset (parameters c.shape) (state pid entry) Bool.false high)
    (pair rest ops (parameters c.shape) (state pid entry) key high)
    (fun i=>c.output (work c.shape pid key i) high)
    (fun i=>component (pairValues c.shape (fun _=>ops) (work c.shape pid key i) (rawValues entry)) high)
    (active_nat c.shape)
    (by intro i hi; have ha : i.1.val<c.shape.half := (active_nat _ _).mp hi
        cases key <;> simp [raw_cacheOffset c hc pid ht entry hm high i ha,Config.output,work])
    (by intro i hi; exact raw_pair rest ops c pid entry key high i ((active_nat _ _).mp hi)) st
  cases key <;> simpa [rotationWrites,work,region,Kind.output,parameters] using h

theorem raw_v_scatter (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2<c.shape.tokens) (entry : Mem α) (hm : MetadataBinding c entry) (st : State α) :
    (scatter (TileShape.allIndices [(parameters c.shape).BLOCK_DIM]) "v_cache"
      (fun i=>i.1.val<(parameters c.shape).HEAD_DIM)
      (valueCacheOffset (parameters c.shape) (state pid entry))
      (fun i=>.mk .real (((state pid entry).mem "v"
        (valueOffset (parameters c.shape) (state pid entry) i)).read .real)) st).mem =
      rawCommit (vWrites c pid (rawValues entry)) st.mem := by
  apply raw_scatter .v c.shape.fullBlock c.shape.width (full_covered c.shape)
    _ _ _ (fun i=>c.shape.cache Bool.false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) i)
    (fun i=>(rawValues entry) .v (c.shape.data .v pid.2 (pid.1/c.shape.groups) i))
  · intro i; rfl
  · intro i _; exact raw_valueCacheOffset c hc pid ht entry hm i
  · intro i _
    rw [valueOffset_nat]
    rfl

theorem raw_written_memory (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (pid : Program) (ht : pid.2<c.shape.tokens) (m : Mem α) (hm : MetadataBinding c m) :
    (written rest ops (parameters c.shape) (state pid m)).mem = rawProgram c ops pid m := by
  let s := state pid m
  let q0 := scatter (TileShape.allIndices [c.shape.halfBlock]) "q" (active (parameters c.shape))
    (dataOffset (parameters c.shape) s Bool.false Bool.false)
    (pair rest ops (parameters c.shape) s Bool.false Bool.false) s
  have hq0 : q0.mem = rawCommit (rotationWrites c ops pid Bool.false Bool.false (rawValues m)) m :=
    raw_rotation_scatter rest ops c hc pid ht m hm Bool.false Bool.false s
  let q1 := scatter (TileShape.allIndices [c.shape.halfBlock]) "q" (active (parameters c.shape))
    (dataOffset (parameters c.shape) s Bool.false Bool.true)
    (pair rest ops (parameters c.shape) s Bool.false Bool.true) q0
  have hq1 : q1.mem = rawCommit (rotationWrites c ops pid Bool.false Bool.true (rawValues m)) q0.mem :=
    raw_rotation_scatter rest ops c hc pid ht m hm Bool.false Bool.true q0
  by_cases leader : pid.1%c.shape.groups=0
  · let k0 := scatter (TileShape.allIndices [c.shape.halfBlock]) "k_cache" (active (parameters c.shape))
      (cacheOffset (parameters c.shape) s Bool.false)
      (pair rest ops (parameters c.shape) s Bool.true Bool.false) q1
    have hk0 : k0.mem = rawCommit (rotationWrites c ops pid Bool.true Bool.false (rawValues m)) q1.mem :=
      raw_rotation_scatter rest ops c hc pid ht m hm Bool.true Bool.false q1
    let k1 := scatter (TileShape.allIndices [c.shape.halfBlock]) "k_cache" (active (parameters c.shape))
      (cacheOffset (parameters c.shape) s Bool.true)
      (pair rest ops (parameters c.shape) s Bool.true Bool.true) k0
    have hk1 : k1.mem = rawCommit (rotationWrites c ops pid Bool.true Bool.true (rawValues m)) k0.mem :=
      raw_rotation_scatter rest ops c hc pid ht m hm Bool.true Bool.true k0
    have hv := raw_v_scatter c hc pid ht m hm k1
    rw [hk1,hk0,hq1,hq0] at hv
    simpa only [written,rawProgram,programWrites,state,parameters,leader,ite_true,
      rawCommit,List.foldl_append] using hv
  · rw [hq0] at hq1
    simpa only [written,rawProgram,programWrites,state,parameters,leader,ite_true,ite_false,
      rawCommit,List.foldl_append,List.foldl_nil] using hq1

theorem written_mem_congr (rest : Algebra α) (ops : Ops α) (p : Parameters)
    (s t : State α) (hm : s.mem = t.mem) (hp : s.pids = t.pids) :
    (written rest ops p s).mem = (written rest ops p t).mem := by
  by_cases leader : s.pids 0%p.KV_GROUP_NUM=0
  all_goals
    have leader' := leader
    rw [hp] at leader'
    simp only [written,leader,leader',ite_true,ite_false]
    simp only [scatter,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
    dataOffset,coefficientOffset,cacheOffset,valueOffset,valueCacheOffset,pair,load,past,block,slot,head,hm,hp]

theorem invocation_memory (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (pid : Program) (ht : pid.2<c.shape.tokens) (s : State α) (hm : MetadataBinding c s.mem) :
    ∃ final, (invocation rest ops c pid).run s = some final ∧ final.mem = rawProgram c ops pid s.mem := by
  let before := (invocation rest ops c pid).initial s
  refine ⟨(execution rest ops (parameters c.shape) before).val,?_,?_⟩
  · exact (execution rest ops (parameters c.shape) before).property
  · rw [execution_memory, written_mem_congr rest ops (parameters c.shape)
      before (state pid s.mem) (by simp [before,state]) (by simp [before,invocation,Invocation.initial,bindRegisters,state])]
    exact raw_written_memory rest ops c hc pid ht s.mem hm

#axiomsClean raw_written_memory
#axiomsClean invocation_memory
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
