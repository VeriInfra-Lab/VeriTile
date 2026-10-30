import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVReal
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.RealBody

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal.QKV
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel BackwardQKVSource
noncomputable section
set_option maxHeartbeats 5000000

def queryTarget (c : Shape) (s : State ℝ) (scale : ℝ) (i : KeyIndex c) : ℝ :=
  ScalarSource.queryTile c (xs c s) scale (s.pids 2) (s.pids 1)
    i.1.val (feature c s i.2.1.val) (history c s "h")
def keyTarget (c : Shape) (s : State ℝ) (scale : ℝ) (i : KeyIndex c) : ℝ :=
  ScalarSource.keyTile c (xs c s) scale (s.pids 2) (s.pids 1)
    i.1.val (feature c s i.2.1.val) (history c s "dh")
def partialTarget (c : Shape) (s : State ℝ) (scale : ℝ) (f : Nat) (i : ValueIndex c) : ℝ :=
  ScalarSource.valuePartial c (xs c s) scale (s.pids 0) (s.pids 2) (s.pids 1)
    i.1.val (column c f i.2.1.val) (history c s "dh")

theorem query_target (c : Shape) (s : State ℝ) (scale : ℝ) (i : KeyIndex c)
    (hi : keyActive c s i) :
    queryValue algebra dot c s scale (trajectory algebra dot c s scale c.valueBlocks).1 i=
      queryTarget c s scale i := by
  rw [query_body]
  exact RealScalar.query_real c (xs c s) scale (s.pids 2) (s.pids 1)
    i.1.val (feature c s i.2.1.val) (history c s "h") hi.1 i.1.isLt hi.2

theorem key_target (c : Shape) (s : State ℝ) (scale : ℝ) (i : KeyIndex c)
    (hc : s.pids 1<c.chunks) (hi : keyActive c s i) :
    keyValue algebra dot c s scale (trajectory algebra dot c s scale c.valueBlocks).1 i=
      keyTarget c s scale i := by
  rw [key_body c s scale i hc]
  exact RealScalar.key_real c (xs c s) scale (s.pids 2) (s.pids 1)
    i.1.val (feature c s i.2.1.val) (history c s "dh") hc
    ((RealScalar.time_lane c (s.pids 1) i.1.val hc i.1.isLt).mp hi.1) hi.2

theorem partial_target (c : Shape) (s : State ℝ) (scale : ℝ) (f : Nat) (i : ValueIndex c)
    (hc : s.pids 1<c.chunks) (hi : valueActive c s f i) :
    partialValue algebra dot c s scale f i=partialTarget c s scale f i := by
  rw [partial_body c s scale f i hc]
  exact RealScalar.value_real c (xs c s) scale (s.pids 0) (s.pids 2) (s.pids 1)
    i.1.val (column c f i.2.1.val) (history c s "dh") hc
    ((RealScalar.time_lane c (s.pids 1) i.1.val hc i.1.isLt).mp hi.1) hi.2

theorem source_result (c : Shape) (s : State ℝ) (scale : ℝ) (hc : s.pids 1<c.chunks) :
    ∃ t, FP.Structural.exec (engine (withDot algebra dot)) (kernel c scale) s=some t ∧
      (∀ i, keyActive c s i → (t.mem "dq" (keyAddress c s i)).read .real=queryTarget c s scale i) ∧
      (∀ i, keyActive c s i → (t.mem "dk" (keyAddress c s i)).read .real=keyTarget c s scale i) ∧
      (∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i →
        (t.mem "dv" (partialAddress c s f i)).read .real=partialTarget c s scale f i) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"dq" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i) →
        (r≠"dk" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i) →
        (r≠"dv" ∨ ∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i → off≠partialAddress c s f i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory algebra dot c scale s
  refine ⟨t,ht,?_,?_,?_,?_⟩
  · intro i hi
    rw [hm,query_readback algebra dot c s scale i hi]
    exact query_target c s scale i hi
  · intro i hi
    rw [hm,key_readback algebra dot c s scale i hi]
    exact key_target c s scale i hc hi
  · intro f hf i hi
    rw [hm,partial_readback algebra dot c s scale f i hf hi]
    exact partial_target c s scale f i hc hi
  · intro r off hq hk hv
    rw [hm]
    exact frame algebra dot c s scale r off hq hk hv

#specNonCircular queryTarget avoiding [run,FP.Structural.exec,BackwardQKVSource.queryValue]
#specNonCircular keyTarget avoiding [run,FP.Structural.exec,BackwardQKVSource.keyValue]
#specNonCircular partialTarget avoiding [run,FP.Structural.exec,BackwardQKVSource.partialValue]
#axiomsClean source_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal.QKV
