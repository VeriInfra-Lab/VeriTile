import bench.repairs.tritonbench_g.rotary_transform.Proofs.HistoricalFPEquiv
import bench.repairs.tritonbench_g.rotary_transform.Proofs.Caller
namespace VeriTile.Bench.Repairs.RotaryTransform.Public
open Native Physical Host

def invoke (call : Call) (views : InputViews) (callerDevice : Nat) (p : Instructions) (initial : Bytes) : Returned Byte :=
  Host.selected call callerDevice p (stageInputs call.shape views call.abi.offsets64 initial)

/-- The ideal source interpretation uses the same initial physical words.
The scalar interpretation is explicit; no unrelated real tensor is assumed. -/
noncomputable def ideal (call : Call) (views : InputViews) (callerDevice : Nat)
    (interpret : Word → ℝ) (initial : Bytes) : Returned ℝ :=
  let staged := stageInputs call.shape views call.abi.offsets64 initial
  mathematical call callerDevice staged (fun r a => interpret (load staged r a))

def nativeExpected (call : Call) (views : InputViews) (p : Instructions) (initial : Bytes)
    (batch row head col : Nat) : Word :=
  nativeFormula (callerConfiguration call.shape call.abi views initial) p
    (callerWords call.shape views initial) batch row head col
noncomputable def realExpected (call : Call) (views : InputViews) (interpret : Word → ℝ) (initial : Bytes)
    (batch row head col : Nat) : ℝ :=
  realFormula (callerConfiguration call.shape call.abi views initial)
    (fun r a => interpret (callerWords call.shape views initial r a)) batch row head col

theorem native_formula_views (c : Config) (s : Shape) (v : InputViews) (wide : Bool)
    (p : Instructions) (m : Bytes) (b r h d : Nat) :
    nativeFormula c p (load (stageInputs s v wide m)) b r h d=
      nativeFormula c p (callerWords s v m) b r h d := by
  rcases staged_inputs s v wide m with ⟨input,cosine,sine⟩
  simp only [nativeFormula,input,cosine,sine]
theorem real_formula_views (c : Config) (s : Shape) (v : InputViews) (wide : Bool)
    (interpret : Word → ℝ) (m : Bytes) (b r h d : Nat) :
    realFormula c (fun r a => interpret (load (stageInputs s v wide m) r a)) b r h d=
      realFormula c (fun r a => interpret (callerWords s v m r a)) b r h d := by
  rcases staged_inputs s v wide m with ⟨input,cosine,sine⟩
  simp only [realFormula,input,cosine,sine]

theorem payload (call : Call) (views : InputViews) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (callerConfiguration call.shape call.abi views initial).Valid)
    (batch row head col : Nat) (hb : batch<call.shape.batches)
    (hr : row<(callerConfiguration call.shape call.abi views initial).length batch)
    (hh : head<call.shape.heads) (hd : col<call.shape.width) (i : Fin 4) :
    (invoke call views callerDevice p initial).memory call.shape.output
      (4*call.shape.address ((callerConfiguration call.shape call.abi views initial).start batch+row) head col+i.val)=
    encode (nativeExpected call views p initial batch row head col) i := by
  have hc' : (originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).Valid := by
    rw [staged_configuration];exact hc
  have hr' : row<(originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).length batch := by
    rw [staged_configuration];exact hr
  have result := correct_native_bytes call callerDevice p (stageInputs call.shape views call.abi.offsets64 initial)
    hc' batch row head col hb hr' hh hd i
  simpa only [invoke,nativeExpected,staged_configuration,native_formula_views] using result

theorem ideal_payload (call : Call) (views : InputViews) (callerDevice : Nat) (interpret : Word → ℝ) (initial : Bytes)
    (hc : (callerConfiguration call.shape call.abi views initial).Valid)
    (batch row head col : Nat) (hb : batch<call.shape.batches)
    (hr : row<(callerConfiguration call.shape call.abi views initial).length batch)
    (hh : head<call.shape.heads) (hd : col<call.shape.width) :
    (ideal call views callerDevice interpret initial).memory call.shape.output
      (call.shape.address ((callerConfiguration call.shape call.abi views initial).start batch+row) head col)=
    realExpected call views interpret initial batch row head col := by
  have hc' : (originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).Valid := by
    rw [staged_configuration];exact hc
  have hr' : row<(originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).length batch := by
    rw [staged_configuration];exact hr
  have result := correct_real call callerDevice (stageInputs call.shape views call.abi.offsets64 initial)
    (fun r a => interpret (load (stageInputs call.shape views call.abi.offsets64 initial) r a))
    hc' batch row head col hb hr' hh hd
  simpa only [ideal,realExpected,staged_configuration,real_formula_views] using result

theorem outside_writes (c : Config) (hc : c.Valid) (r : Region) (a : Nat)
    (outside : r≠c.shape.output ∨ size c.shape≤a) :
    ∀ w,c.Active w → ¬Writes c w r a := by
  intro w hw writes
  rcases writes with ⟨region,lo|hi⟩
  · have bound := cell_bound c hc w hw false
    rcases outside with outside|outside <;> simp_all <;> omega
  · have bound := cell_bound c hc w hw true
    rcases outside with outside|outside <;> simp_all <;> omega

theorem outside_copy (s : Shape) (r : Region) (a : Nat)
    (outside : r≠s.output ∨ size s≤a) : ¬copied s r a := by
  intro copy
  rcases copy with ⟨hi,hr,ha,_⟩
  rcases outside with outside|outside
  · simp [Shape.output,hi,hr] at outside
  · omega

/-- Every caller byte outside the output allocation's logical extent is
preserved. Caller CU, coefficient/offset backing storage and padding are
included; only named private normalized temporaries are excluded. -/
theorem frame (call : Call) (views : InputViews) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (callerConfiguration call.shape call.abi views initial).Valid)
    (r : Region) (b : Nat) (caller : CallerRegion r)
    (outside : r≠call.shape.output ∨ size call.shape≤b/4) :
    (invoke call views callerDevice p initial).memory r b=initial r b := by
  have hc' : (originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).Valid := by
    rw [staged_configuration];exact hc
  unfold invoke
  rw [caller_frame call callerDevice p _ hc' r b caller.2.2.1 (outside_copy call.shape r (b/4) outside)]
  · exact stage_frame call.shape views call.abi.offsets64 initial r b caller
  · exact outside_writes _ hc' r (b/4) outside

theorem source_interpretation (call : Call) (views : InputViews) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (callerConfiguration call.shape call.abi views initial).Valid) :
    load (invoke call views callerDevice p initial).memory=
    Native.project (source (callerConfiguration call.shape call.abi views initial) (Native.ops p)
      (Native.liftMemory (load (prepareBytes call.shape (normalizeFor call.shape call.abi
        (stageInputs call.shape views call.abi.offsets64 initial)))))) := by
  have hc' : (originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).Valid := by
    rw [staged_configuration];exact hc
  unfold invoke
  rw [selected_memory call callerDevice p _ hc',native_source _ hc',staged_configuration]

theorem return_contract (call : Call) (views : InputViews) (callerDevice : Nat) (p : Instructions) (initial : Bytes) :
    (invoke call views callerDevice p initial).valueIdentity=(if call.shape.inplace then call.inputIdentity else call.freshIdentity) ∧
    (invoke call views callerDevice p initial).valueRegion=call.shape.output ∧
    (invoke call views callerDevice p initial).shape=call.shape ∧
    (invoke call views callerDevice p initial).fp32=true ∧
    (invoke call views callerDevice p initial).device=call.device ∧
    (invoke call views callerDevice p initial).deviceRestored=callerDevice := returned_tensor ..

theorem fp_equivalent (call : Call) (views : InputViews) (callerDevice : Nat)
    (actual target : Instructions) (ports : Obligations actual target) (initial : Bytes) :
    invoke call views callerDevice actual initial=invoke call views callerDevice target initial := by
  rw [instructions_eq actual target ports]

#specNonCircular nativeExpected avoiding [invoke,Host.selected,Native.graph,Native.pairWord,Physical.native]
#specNonCircular realExpected avoiding [ideal,source,pairValues,sourcePair,Host.mathematical]
#axiomsClean payload
#axiomsClean ideal_payload
#axiomsClean frame
#axiomsClean source_interpretation
#axiomsClean return_contract
#axiomsClean fp_equivalent
end VeriTile.Bench.Repairs.RotaryTransform.Public
