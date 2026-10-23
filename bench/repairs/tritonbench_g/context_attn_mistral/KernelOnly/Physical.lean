import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Grid
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Native
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.Execution

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelPhysical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API)
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
open Checked NativeIndex Execution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

abbrev Base := RegionName → Nat

/-- Read-only source roles retain the actual physical caller addresses. -/
structure Binding (c : Host.Call) (buffers : Buffers) (base : Base) : Prop where
  query : base c.regions.query=(buffers.data .query).base
  key : base c.regions.key=(buffers.data .key).base
  value : base c.regions.value=(buffers.data .value).base
  output : base c.regions.out=(buffers.data .output).base
  start : base c.regions.start=(buffers.metadataBuffer .start).base
  length : base c.regions.sequence=(buffers.metadataBuffer .length).base
  floating : ∀ rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out],
    rg∉[c.regions.start,c.regions.sequence]
  outputSeparate : ∀ rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence],c.regions.out≠rg

def initial (c : Host.Call) (api : API Word Store Byte) (base : Base) (before : Store) : State NativePhysical.Expr where
  mem := fun rg off => if rg∈[c.regions.start,c.regions.sequence]
    then .mk .int (api.read before (base rg+4*off)).toInt
    else .mk .real (.leaf .fp32 (api.read before (base rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := c.grid.toNumPids
  undef := fun dtype _ _ => defaultValue dtype

def Observed (c : Host.Call) (api : API Word Store Byte) (base : Base) (before : Store) : Prop :=
  (∀ b,b<c.batches → (api.read before (base c.regions.start+4*b)).toInt=(c.start b : Int)) ∧
  (∀ b,b<c.batches → (api.read before (base c.regions.sequence+4*b)).toInt=(c.length b : Int))

def reference (c : Host.Call) (api : API Word Store Byte) (base : Base) (before : Store) : State NativePhysical.Expr :=
  KernelReference.grid c (KernelNative.backend c.layout.D c.layout.window.toNat) (KernelReference.allPrograms c) (initial c api base before)

def word (c : Host.Call) (ops : ScalarOps) (graph : State NativePhysical.Expr) (q : Coord c) : Word :=
  let e := (graph.mem c.regions.out (offset q)).read .real
  KernelNative.evaluate ops c.layout.D c.layout.window.toNat e (KernelNative.owner c.layout.D e)

/-- Initial caller words feed a source-independent tile recurrence and fixed
instruction profile. No source execution or supplied output occurs here. -/
def target (c : Host.Call) (api : API Word Store Byte) (base : Base) (ops : ScalarOps)
    (before : Store) (q : Coord c) : Word := word c ops (reference c api base before) q

def stores (c : Host.Call) (api : API Word Store Byte) (buffers : Buffers) (ops : ScalarOps)
    (graph : State NativePhysical.Expr) (before : Store) : Store :=
  (all c).foldl (fun memory q => api.write memory (address buffers q) (word c ops graph q)) before

structure Result (Store : Type) where
  graph : State NativePhysical.Expr
  memory : Store

/-- The physical stores consume the actual complete source grid's final
output cells. The recurrence is used only by the separate target. -/
def run (c : Host.Call) (api : API Word Store Byte) (base : Base) (buffers : Buffers)
    (ops : ScalarOps) (before : Store) : Option (Result Store) := do
  let graph ← KernelReference.runGrid c (KernelNative.backend c.layout.D c.layout.window.toNat)
    (KernelReference.allPrograms c) (initial c api base before)
  return ⟨graph,stores c api buffers ops graph before⟩

theorem words (c : Host.Call) (api : API Word Store Byte) (base : Base) (buffers : Buffers)
    (ops : ScalarOps) (tokens : Nat) (request : Request)
    (accepted : check c tokens request buffers=Bool.true) (before : Store) :
    ∃ result,run c api base buffers ops before=some result ∧
      KernelReference.runGrid c (KernelNative.backend c.layout.D c.layout.window.toNat) (KernelReference.allPrograms c)
        (initial c api base before)=some result.graph ∧
      result.graph.mem=(reference c api base before).mem ∧
      (∀ q,api.read result.memory (address buffers q)=target c api base ops before q) ∧
      (∀ i,Outside buffers (all c) i → api.byte result.memory i=api.byte before i) := by
  have admitted := check_iff.mp accepted
  obtain ⟨graph,source,equal⟩ := KernelReference.whole c (KernelNative.backend c.layout.D c.layout.window.toNat)
    (KernelReference.allPrograms c) (initial c api base before)
  refine ⟨⟨graph,stores c api buffers ops graph before⟩,?_,source,equal,?_,?_⟩
  · simp [run,source]
  · intro q
    have step := readback api 0 admitted (fun _ q => word c ops graph q) (all c) before q (coverage q)
    change api.read (execute api 0 c tokens buffers (fun _ q => word c ops graph q) (all c) before) (address buffers q)=_
    rw [step]
    simp only [word,target,equal,reference]
  · intro i outside
    change api.byte (execute api 0 c tokens buffers (fun _ q => word c ops graph q) (all c) before) i=_
    exact byte_frame api 0 admitted _ _ _ _ outside

theorem empty_stores (c : Host.Call) (api : API Word Store Byte) (buffers : Buffers)
    (ops : ScalarOps) (graph : State NativePhysical.Expr) (before : Store)
    (empty : ∀ q : Coord c,False) : stores c api buffers ops graph before=before := by
  have all_empty : all c=[] := List.eq_nil_iff_forall_not_mem.mpr (fun q _ => empty q)
  simp [stores,all_empty,execute]

#specNonCircular target avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.program,KernelReference.runGrid,run,stores]
#axiomsClean words
#axiomsClean empty_stores
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelPhysical
