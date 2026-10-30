import bench.repairs.tritonbench_g.chunk_linear_attn.HostAPI
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

/-- The default is Python's binary64 power result; the device scalar is its
actual FP32 ABI conversion. These are scalar operations, not tensor answers. -/
structure ScalarABI (decode32 : BitVec 32 → ℝ) (decode64 : BitVec 64 → ℝ)
    (powerDefault : Nat → BitVec 64) (convert32 : BitVec 64 → BitVec 32) (c : Invocation) : Prop where
  origin : match c.scale.origin with
    | .supplied x => decode64 c.scale.pythonBinary64=x
    | .defaultPower k => c.scale.pythonBinary64=powerDefault k
  conversion : c.scale.argumentFP32=convert32 c.scale.pythonBinary64
  interpretation : c.scale.scalarValue=decode32 c.scale.argumentFP32

def buffer (r : RegionName) (t : Tensor) : Physical.Buffer := ⟨r,t.base,t.shape.prod⟩
def outputBuffers (a : Allocations) : List Physical.Buffer :=
  [buffer "history" a.history,buffer "output" a.output,buffer "historyGradient" a.reverseHistory,
   buffer "queryGradient" a.queryGradient,buffer "keyGradient" a.keyGradient,
   buffer "valuePartials" a.valuePartials,buffer "valueGradient" a.valueGradient]++
    (a.finalState.toList.map (buffer "final"))++(a.initialGradient.toList.map (buffer "initialGradient"))
def protectedTensors (c : Invocation) (a : Allocations) : List Tensor :=
  c.forwardInputs++c.incoming.toList++c.terminal.toList++[a.incomingDense]++a.terminalDense.toList
def inputBuffers (c : Invocation) (a : Allocations) : List Physical.Buffer :=
  (protectedTensors c a).map (fun t => ⟨"protectedInput",t.base,span t⟩)

/-- Successful physical allocation and pointer-room facts for the exact tensor
descriptors returned by this call. It contains no kernel or numerical premise. -/
structure PhysicalAllocation (c : Invocation) (a : Allocations) : Prop where
  separate : ∀ x∈outputBuffers a, ∀ y∈outputBuffers a, x≠y → Physical.Separate x y
  inputFresh : ∀ x∈inputBuffers c a, ∀ y∈outputBuffers a, Physical.Separate x y
  pointerRoom : PointerABI c.dims (outputBuffers a++inputBuffers c a)

def publish {α : Type} (a : Allocations) (values : RegionName → Nat → α) (before : Nat → α) : Nat → α :=
  Physical.publish (outputBuffers a) values before

theorem output_regions (c : Invocation) (a : Allocations) (ac : AllocationContract c a) :
    (outputBuffers a).map Physical.Buffer.region=Physical.outputRegions c.initial.isSome c.final := by
  rw [← ac.finalFlag,← ac.initialFlag]
  cases hf : a.finalState <;> cases hi : a.initialGradient <;> simp [outputBuffers,Physical.outputRegions,buffer,hf,hi]

theorem published_read {α : Type} (c : Invocation) (a : Allocations) (pa : PhysicalAllocation c a)
    (values : RegionName → Nat → α) (before : Nat → α) (r : RegionName) (t : Tensor)
    (ht : buffer r t∈outputBuffers a) (o : Nat) (ho : o<t.shape.prod) :
    publish a values before (t.base+o)=values r o :=
  Physical.publish_read (outputBuffers a) values before (buffer r t) ht o ho
    (fun y hy hn => pa.separate _ ht y hy (Ne.symm hn))

theorem published_inputs {α : Type} (c : Invocation) (a : Allocations) (pa : PhysicalAllocation c a)
    (values : RegionName → Nat → α) (before : Nat → α) (t : Tensor)
    (ht : t∈protectedTensors c a) (o : Nat) (ho : o<span t) :
    publish a values before (t.base+o)=before (t.base+o) := by
  exact Physical.publish_input (outputBuffers a) values before ⟨"protectedInput",t.base,span t⟩
    (pa.inputFresh _ (List.mem_map.mpr ⟨t,ht,rfl⟩)) o ho

theorem published_frame {α : Type} (a : Allocations) (values : RegionName → Nat → α) (before : Nat → α)
    (address : Nat) (h : ∀ b∈outputBuffers a, ¬Physical.Contains b address) :
    publish a values before address=before address :=
  Physical.publish_frame (outputBuffers a) values before address h

theorem output_mem (a : Allocations) : buffer "output" a.output∈outputBuffers a := by simp [outputBuffers]
theorem query_mem (a : Allocations) : buffer "queryGradient" a.queryGradient∈outputBuffers a := by simp [outputBuffers]
theorem key_mem (a : Allocations) : buffer "keyGradient" a.keyGradient∈outputBuffers a := by simp [outputBuffers]
theorem value_mem (a : Allocations) : buffer "valueGradient" a.valueGradient∈outputBuffers a := by simp [outputBuffers]
theorem final_mem (a : Allocations) (t : Tensor) (h : a.finalState=some t) : buffer "final" t∈outputBuffers a := by simp [outputBuffers,h]
theorem initial_mem (a : Allocations) (t : Tensor) (h : a.initialGradient=some t) : buffer "initialGradient" t∈outputBuffers a := by simp [outputBuffers,h]

theorem packed_value (c : Invocation) (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V) :
    (head*c.dims.T+t.val)*c.dims.V+j.val<(valueShape c.dims).prod := by
  have h := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
    (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh t.isLt) j.isLt
  simpa only [valueShape,List.prod_cons,List.prod_nil,Nat.mul_one,Nat.mul_assoc] using h
 theorem packed_query (c : Invocation) (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (e : Fin c.dims.K) :
    (head*c.dims.T+t.val)*c.dims.K+e.val<(queryShape c.dims).prod := by
  have h := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
    (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh t.isLt) e.isLt
  simpa only [queryShape,List.prod_cons,List.prod_nil,Nat.mul_one,Nat.mul_assoc] using h
 theorem packed_state (c : Invocation) (head : Nat) (hh : head<c.dims.B*c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V) :
    (head*c.dims.K+e.val)*c.dims.V+j.val<(stateShape c.dims).prod := by
  have h := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
    (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh e.isLt) j.isLt
  simpa only [stateShape,List.prod_cons,List.prod_nil,Nat.mul_one,Nat.mul_assoc] using h

theorem absent_initial (c : Invocation) (a : Allocations) (ac : AllocationContract c a) (h : c.initial=none) :
    a.initialGradient=none ∧ (returnedBackward a)[4]?=some none := by
  have hf := ac.initialFlag
  simp only [h,Option.isSome_none] at hf
  have ha : a.initialGradient=none := by
    cases hg : a.initialGradient
    · rfl
    · simp [hg] at hf
  simp [returnedBackward,ha]
 theorem absent_final (c : Invocation) (a : Allocations) (ac : AllocationContract c a) (h : c.final=Bool.false) :
    (returnedForward a).2=none := by
  have hf := ac.finalFlag.trans h
  cases hg : a.finalState
  · exact hg
  · simp [hg] at hf
 theorem absent_seed_value (c : Invocation) (m : Memory ℝ) (head e j : Nat) (h : c.initial=none) :
    Real.endpoint c.mathCall.forward head m Bool.false c.initial.isSome e j=0 := by
  simp [Real.endpoint,h]
 theorem absent_terminal_value (c : Invocation) (m : Memory ℝ) (head e j : Nat) (h : c.terminal=none) :
    Real.endpoint c.mathCall.backward head m Bool.true c.terminal.isSome e j=0 := by
  simp [Real.endpoint,h]
 theorem incoming_copy (c : Invocation) (a : Allocations) (physical : Nat → Cell ℝ) (fallback : Memory ℝ)
    (copies : CotangentCopies Real.algebra c a physical)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V) :
    Real.incoming c.mathCall.backward (b.val*c.dims.H+h.val) (logical c a physical fallback) t.val j.val=
      (match c.incoming with
       | none => 0
       | some src => (physical (src.base+offset4 src b.val h.val t.val j.val)).read .real) := by
  unfold Real.incoming logical descriptor
  have he : (b.val*c.dims.H+h.val)*(c.dims.T*c.dims.V)+t.val*c.dims.V+j.val=
      ((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.V+j.val := by ring
  change (physical (a.incomingDense.base+((b.val*c.dims.H+h.val)*(c.dims.T*c.dims.V)+t.val*c.dims.V+j.val))).read .real=_
  rw [he,← Nat.add_assoc,copies.incoming]
  cases c.incoming <;> rfl

#axiomsClean published_read
#axiomsClean published_inputs
#axiomsClean published_frame
#axiomsClean incoming_copy
#axiomsClean absent_initial
#specNonCircular ScalarABI avoiding [execute,Real.Call.result,Real.Call.target]
#specNonCircular PhysicalAllocation avoiding [execute,publish,Real.Call.result]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host
