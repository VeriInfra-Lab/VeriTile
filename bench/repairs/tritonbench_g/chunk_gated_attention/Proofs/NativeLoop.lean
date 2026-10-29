import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeBody

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

abbrev Reference (c : Shape) := (HIndex c→NativeGraph.Expr) × (RegionName→Nat→Cell NativeGraph.Expr)

/-- Independent arithmetic recurrence paired with the explicit history stores.
All arithmetic reads the initial input snapshot; only the fresh history changes. -/
def referenceStep (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (gateK : Bool)
    (chunk : Nat) (a : Reference c) : Reference c :=
  (update c base s gateK chunk a.1,
    (historyWrite c base s {s with mem:=a.2} chunk a.1).mem)

def reference (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (gateK hasSeed : Bool) : Reference c :=
  foldRange (referenceStep c base s gateK) 0 c.NT 1 (initial algebra c base s hasSeed,s.mem)

theorem historyWrite_memory (c : Shape) (base : Nat) (initial a b : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c→NativeGraph.Expr) (same : a.mem=b.mem) :
    (historyWrite c base initial a chunk acc).mem=(historyWrite c base initial b chunk acc).mem := by
  simp only [historyWrite,scatter_memory,same]

theorem entryMemory (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool) (s : State NativeGraph.Expr) :
    (entryExecution algebra c base gateK hasSeed storeFinal s).val.mem=s.mem := by
  cases hasSeed <;> simp [entryExecution]

theorem loopExecution (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (ctx : Context c base initial current)
    (ha : HasAcc c current (NativeValue.initial algebra c base initial hasSeed))
    (hm : current.mem=initial.mem) :
    ∃ t,FP.Structural.range (engine algebra) "i_t" 0 c.NT 1
      (parts c base gateK hasSeed storeFinal).body current=some t ∧
      Context c base initial t ∧ HasAcc c t (reference c base initial gateK hasSeed).1 ∧
      t.mem=(reference c base initial gateK hasSeed).2 := by
  let I:=fun (_ : Nat) (a : Reference c) (s : State NativeGraph.Expr)=>
    Context c base initial s ∧ HasAcc c s a.1 ∧ s.mem=a.2
  apply range_values (engine algebra) "i_t" (parts c base gateK hasSeed storeFinal).body
    (referenceStep c base initial gateK) I
    (fun a s=>Context c base initial s ∧ HasAcc c s a.1 ∧ s.mem=a.2)
    0 c.NT 1 (NativeValue.initial algebra c base initial hasSeed,initial.mem) current
  · intro n a s h hn
    let t:=bodyExecution c base gateK hasSeed storeFinal initial s n a.1 h.1 h.2.1
    refine ⟨t.val,t.property,bodyContext c base gateK hasSeed storeFinal initial s n a.1 h.1 h.2.1,
      bodyAcc c base gateK hasSeed storeFinal initial s n a.1 h.1 h.2.1,?_⟩
    exact (bodyMemory c base gateK hasSeed storeFinal initial s n a.1 h.1 h.2.1).trans
      (historyWrite_memory c base initial s {initial with mem:=a.2} n a.1 h.2.2)
  · intro n a s h
    exact h
  · exact ⟨ctx,ha,hm⟩

#specNonCircular reference avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.range]
#axiomsClean loopExecution
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
