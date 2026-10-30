import bench.repairs.tritonbench_g.chunk_linear_attn.HostIngress
import bench.repairs.tritonbench_g.chunk_linear_attn.BoundMeaningGrid
import bench.repairs.tritonbench_g.chunk_linear_attn.KernelAddresses
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure PublishedResult (α : Type) where
  words : Nat → α
  forward : Host.Tensor × Option Host.Tensor
  backward : List (Option Host.Tensor)
  context : Host.Context

def realPublished (c : Host.Invocation) (a : Host.Allocations) (before : Nat → Cell ℝ) (fallback : Memory ℝ) :
    PublishedResult (Cell ℝ) :=
  ⟨Host.publish a (c.mathCall.result (Host.logical c a before fallback)) before,
    Host.returnedForward a,Host.returnedBackward a,Host.saveContext c a⟩
def callerGradient (c : Host.Invocation) (before : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H) :=
  Real.gradients c.scale.scalarValue (Host.callerInput c before b h)
    (Host.callerIncoming c before b h) (Host.callerTerminal c before b h)

structure RealPost (c : Host.Invocation) (a : Host.Allocations) (before : Nat → Cell ℝ)
    (fallback : Memory ℝ) (r : PublishedResult (Cell ℝ)) : Prop where
  forward : r.forward=(a.output,a.finalState)
  backward : r.backward=[some a.queryGradient,some a.keyGradient,some a.valueGradient,none,a.initialGradient,none]
  context : r.context=Host.saveContext c a
  source_execution : ∃ result, Host.execute Real.algebra Real.dotPrimitive c a (Host.logical c a before fallback)=some result ∧
    result.memory=c.mathCall.result (Host.logical c a before fallback)
  output : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V),
    r.words (a.output.base+((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.V+j.val)=
      .mk .real (Real.output c.scale.scalarValue (Host.callerInput c before b h) t j)
  query : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (e : Fin c.dims.K),
    r.words (a.queryGradient.base+((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.K+e.val)=
      .mk .real ((callerGradient c before b h).q t e)
  key : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (e : Fin c.dims.K),
    r.words (a.keyGradient.base+((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.K+e.val)=
      .mk .real ((callerGradient c before b h).k t e)
  value : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V),
    r.words (a.valueGradient.base+((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.V+j.val)=
      .mk .real ((callerGradient c before b h).v t j)
  final : ∀ dst, a.finalState=some dst → ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V),
    r.words (dst.base+((b.val*c.dims.H+h.val)*c.dims.K+e.val)*c.dims.V+j.val)=
      .mk .real (Real.finalState (Host.callerInput c before b h) e j)
  initial : ∀ dst, a.initialGradient=some dst → ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V),
    r.words (dst.base+((b.val*c.dims.H+h.val)*c.dims.K+e.val)*c.dims.V+j.val)=
      .mk .real ((callerGradient c before b h).initial e j)
  vjp : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H),
    Real.IsVJP c.scale.scalarValue (Host.callerInput c before b h) (Host.callerIncoming c before b h)
      (Host.callerTerminal c before b h) (callerGradient c before b h)
  inputs : ∀ src∈Host.protectedTensors c a, ∀ o, o<Host.span src → r.words (src.base+o)=before (src.base+o)
  frame : ∀ address, (∀ buf∈Host.outputBuffers a, ¬Physical.Contains buf address) → r.words address=before address

theorem real_read (c : Host.Invocation) (a : Host.Allocations) (pa : Host.PhysicalAllocation c a)
    (before : Nat → Cell ℝ) (fallback : Memory ℝ) (name : RegionName) (dst : Host.Tensor)
    (hd : Host.buffer name dst∈Host.outputBuffers a) (o : Nat) (ho : o<dst.shape.prod) :
    (realPublished c a before fallback).words (dst.base+o)=c.mathCall.result (Host.logical c a before fallback) name o :=
  Host.published_read c a pa _ before name dst hd o ho

theorem real_complete (c : Host.Invocation) (cc : Host.Contract c) (a : Host.Allocations)
    (ac : Host.AllocationContract c a) (pa : Host.PhysicalAllocation c a) (initial : Host.InitialLayout c)
    (before : Nat → Cell ℝ) (fallback : Memory ℝ) (copies : Host.CotangentCopies Real.algebra c a before) :
    RealPost c a before fallback (realPublished c a before fallback) := by
  refine ⟨rfl,rfl,rfl,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact ⟨⟨c.mathCall.result (Host.logical c a before fallback),Host.returnedForward a,Host.returnedBackward a,Host.saveContext c a⟩,
      Host.execute_complete Real.algebra Real.dotPrimitive c a _,rfl⟩
  · intro b h t j
    have hh := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt b.isLt h.isLt
    have hb := Host.packed_value c _ hh t j
    rw [← ac.outputShape] at hb
    have hr := real_read c a pa before fallback "output" a.output (Host.output_mem a) _ hb
    simp only [← Nat.add_assoc] at hr
    have hm := c.mathCall.output_read cc.valid (Host.logical c a before fallback) (b.val*c.dims.H+h.val) hh t j
    simp only [show c.mathCall.dims=c.dims from rfl,show c.mathCall.scale=c.scale.scalarValue from rfl,show c.mathCall.seed=c.initial.isSome from rfl] at hm
    rw [hr,hm]
    have hi : Real.inputs c.mathCall.forward (b.val*c.dims.H+h.val) (Host.logical c a before fallback) c.initial.isSome=Host.callerInput c before b h :=
      Host.caller_inputs c cc a initial before fallback b h
    rw [hi]
    rfl
  · intro b h t e
    have hh := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt b.isLt h.isLt
    have hb := Host.packed_query c _ hh t e
    rw [← ac.queryShape] at hb
    have hr := real_read c a pa before fallback "queryGradient" a.queryGradient (Host.query_mem a) _ hb
    simp only [← Nat.add_assoc] at hr
    have hm := c.mathCall.query_read cc.valid (Host.logical c a before fallback) (b.val*c.dims.H+h.val) hh t e
    simp only [show c.mathCall.dims=c.dims from rfl,show c.mathCall.scale=c.scale.scalarValue from rfl,show c.mathCall.seed=c.initial.isSome from rfl] at hm
    rw [hr,hm]
    rw [Host.caller_target c cc a ac initial before fallback copies b h]
    rfl
  · intro b h t e
    have hh := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt b.isLt h.isLt
    have hb := Host.packed_query c _ hh t e
    rw [← ac.keyShape] at hb
    have hr := real_read c a pa before fallback "keyGradient" a.keyGradient (Host.key_mem a) _ hb
    simp only [← Nat.add_assoc] at hr
    have hm := c.mathCall.key_read cc.valid (Host.logical c a before fallback) (b.val*c.dims.H+h.val) hh t e
    simp only [show c.mathCall.dims=c.dims from rfl,show c.mathCall.scale=c.scale.scalarValue from rfl,show c.mathCall.seed=c.initial.isSome from rfl] at hm
    rw [hr,hm]
    rw [Host.caller_target c cc a ac initial before fallback copies b h]
    rfl
  · intro b h t j
    have hh := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt b.isLt h.isLt
    have hb := Host.packed_value c _ hh t j
    rw [← ac.valueShape] at hb
    have hr := real_read c a pa before fallback "valueGradient" a.valueGradient (Host.value_mem a) _ hb
    simp only [← Nat.add_assoc] at hr
    have hm := c.mathCall.value_read cc.valid (Host.logical c a before fallback) (b.val*c.dims.H+h.val) hh t j
    simp only [show c.mathCall.dims=c.dims from rfl,show c.mathCall.scale=c.scale.scalarValue from rfl,show c.mathCall.seed=c.initial.isSome from rfl] at hm
    rw [hr,hm]
    rw [Host.caller_target c cc a ac initial before fallback copies b h]
    rfl
  · intro dst hd b h e j
    have hh := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt b.isLt h.isLt
    have hs := ac.stateShapes dst (by simp [hd])
    have hb := Host.packed_state c _ hh e j
    rw [← hs] at hb
    have hr := real_read c a pa before fallback "final" dst (Host.final_mem a dst hd) _ hb
    simp only [← Nat.add_assoc] at hr
    have hf := ac.finalFlag
    have hf' : c.mathCall.final=Bool.true := by simpa only [hd,Option.isSome_some] using hf.symm
    have hm := c.mathCall.final_read cc.valid (Host.logical c a before fallback) hf' (b.val*c.dims.H+h.val) hh e j
    simp only [show c.mathCall.dims=c.dims from rfl,show c.mathCall.seed=c.initial.isSome from rfl] at hm
    rw [hr,hm]
    have hi : Real.inputs c.mathCall.forward (b.val*c.dims.H+h.val) (Host.logical c a before fallback) c.initial.isSome=Host.callerInput c before b h :=
      Host.caller_inputs c cc a initial before fallback b h
    rw [hi]
    rfl
  · intro dst hd b h e j
    have hh := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt b.isLt h.isLt
    have hs := ac.stateShapes dst (by simp [hd])
    have hb := Host.packed_state c _ hh e j
    rw [← hs] at hb
    have hr := real_read c a pa before fallback "initialGradient" dst (Host.initial_mem a dst hd) _ hb
    simp only [← Nat.add_assoc] at hr
    have hf := ac.initialFlag
    have hf' : c.mathCall.seed=Bool.true := by simpa only [hd,Option.isSome_some] using hf.symm
    have hm := c.mathCall.initial_read cc.valid (Host.logical c a before fallback) hf' (b.val*c.dims.H+h.val) hh e j
    simp only [show c.mathCall.dims=c.dims from rfl,show c.mathCall.seed=c.initial.isSome from rfl] at hm
    rw [hr,hm]
    rw [Host.caller_target c cc a ac initial before fallback copies b h]
    rfl
  · intro b h
    exact Real.gradients_correct _ _ _ _
  · intro src hs o ho
    exact Host.published_inputs c a pa _ before src hs o ho
  · intro address h
    exact Host.published_frame a _ before address h

def regionBits (c : Host.Invocation) (a : Host.Allocations) (physical : Nat → BitVec 32)
    (fallback : RegionName → Nat → BitVec 32) : RegionName → Nat → BitVec 32 :=
  fun r o => match Host.descriptor c a r with | some t => physical (t.base+o) | none => fallback r o

def symbolicInvocation (c : Host.Invocation) : Host.Invocation :=
  {c with scale:={c.scale with scalarValue:=1}}

def nativePublished (I : ISA (BitVec 32)) (c : Host.Invocation) (a : Host.Allocations)
    (before : Nat → BitVec 32) (fallback : RegionName → Nat → BitVec 32) : PublishedResult (BitVec 32) :=
  ⟨Host.publish a (fun r o => (NativeBound.graph c.scale.argumentFP32 c.mathCall (regionBits c a before fallback) r o).eval I) before,
    Host.returnedForward a,Host.returnedBackward a,Host.saveContext c a⟩

structure NativePost (profile : NativeTerms.FP32Profile) (c : Host.Invocation) (a : Host.Allocations)
    (before : Nat → BitVec 32) (fallback : RegionName → Nat → BitVec 32) (result : PublishedResult (BitVec 32)) : Prop where
  forward : result.forward=(a.output,a.finalState)
  backward : result.backward=[some a.queryGradient,some a.keyGradient,some a.valueGradient,none,a.initialGradient,none]
  context : result.context=Host.saveContext c a
  source_execution : ∃ sourceResult, Host.execute NativeTerms.algebra NativeTerms.dotPrimitive (symbolicInvocation c) a
      (NativeTerms.inputMemory (regionBits c a before fallback))=some sourceResult ∧
    sourceResult.memory=NativeBound.rawMemory c.mathCall (regionBits c a before fallback)
  outputs : ∀ buf∈Host.outputBuffers a, ∀ o, o<buf.words → result.words (buf.base+o)=
    (NativeBound.graph c.scale.argumentFP32 c.mathCall (regionBits c a before fallback) buf.region o).eval profile.isa
  inputs : ∀ src∈Host.protectedTensors c a, ∀ o, o<Host.span src → result.words (src.base+o)=before (src.base+o)
  frame : ∀ address, (∀ buf∈Host.outputBuffers a, ¬Physical.Contains buf address) → result.words address=before address

/-- Complete native word-memory publication with all four source grids and
explicit dV host sum. Deferred premises mention only six scalar FP32 leaves. -/
theorem native_complete (I : ISA (BitVec 32)) (profile : NativeTerms.FP32Profile)
    (numeric : NativeTerms.PrimitiveLeaves I profile) (c : Host.Invocation) (cc : Host.Contract c) (a : Host.Allocations)
    (ac : Host.AllocationContract c a) (pa : Host.PhysicalAllocation c a) (before : Nat → BitVec 32) (fallback : RegionName → Nat → BitVec 32) :
    NativePost profile c a before fallback (nativePublished I c a before fallback) := by
  refine ⟨rfl,rfl,rfl,?_,?_,?_,?_⟩
  · exact ⟨⟨NativeBound.rawMemory c.mathCall (regionBits c a before fallback),Host.returnedForward a,Host.returnedBackward a,Host.saveContext (symbolicInvocation c) a⟩,
      Host.execute_complete NativeTerms.algebra NativeTerms.dotPrimitive (symbolicInvocation c) a _,rfl⟩
  · intro buf hb o ho
    change Host.publish a _ before (buf.base+o)=_
    unfold Host.publish
    rw [Physical.publish_read (Host.outputBuffers a) _ before buf hb o ho
      (fun y hy hn => pa.separate buf hb y hy (Ne.symm hn))]
    exact NativeBound.graph_fp c.scale.argumentFP32 I profile numeric c.mathCall _ buf.region o
  · intro src hs o ho
    exact Host.published_inputs c a pa _ before src hs o ho
  · intro address h
    exact Host.published_frame a _ before address h

#axiomsClean real_complete
#axiomsClean native_complete
#specNonCircular callerGradient avoiding [realPublished,Real.Call.result,executeFour,gridMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
