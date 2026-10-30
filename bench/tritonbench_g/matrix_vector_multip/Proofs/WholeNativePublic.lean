import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeNativeHost
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeContract

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholePublic
open VeriTile VeriTile.Triton FP.Structural WholeSource WholeGrid WholeHost WholeFinite
noncomputable section
set_option maxHeartbeats 6000000

variable {α : Type} {x : Inputs} {entry : Nat → α} {preexisting : Nat → Prop}

theorem native_outcomes {entry : Nat → BitVec 32} (B : WholeNative.Scalars)
    (graphs : (c : Config) → Nat → WholeNative.Graph c.blockRows c.blockColumns)
    (rt : Runtime (BitVec 32) x entry preexisting) :
    Outcomes (fun a i => WholeNativeHost.nativeFormula B graphs (rt.plan a).selected.config x entry i.val)
      rt (WholeNativeHost.execute B graphs rt) := by
  by_cases compatible : x.columns=x.vectorLength
  · cases allocated : rt.allocate with
    | none => simp [WholeNativeHost.execute,compatible,allocated,Outcomes]
    | some a =>
      cases ready : rt.launchReady a with
      | «false» =>
        simp [WholeNativeHost.execute,compatible,allocated,ready,Outcomes]
      | «true» =>
        obtain ⟨out,ran,shape,dtype,device,stride,base,values,exterior,frame⟩ :=
          WholeNativeHost.execute_returns B graphs rt compatible a allocated ready
        rw [ran]
        refine ⟨compatible,a,allocated,ready,shape,dtype,device,stride,base,?_,exterior,frame⟩
        intro i
        rw [values i.val i.isLt,WholeNativeHost.nativeRow_entry B graphs _ compatible]
  · simp [WholeNativeHost.execute,compatible,Outcomes]

/-- Complete actual ordered FP32 instruction graph on original entry words.
Only scalar FMA/add conformance remains numerical. Graphs contain no payload
or output callbacks, and their completeness is a finite structural certificate.
The external compiler/model binding is checked per exact specialization. -/
specification mv_native_fp32_source_outcomes {entry : Nat → BitVec 32} (B : WholeNative.Scalars)
    (graphs : (c : Config) → Nat → WholeNative.Graph c.blockRows c.blockColumns)
    (rt : Runtime (BitVec 32) x entry preexisting)
    (finite : ∀ a,rt.allocate=some a → rt.launchReady a=Bool.true → FiniteAdmission x entry preexisting a) :
    Outcomes (fun a i => WholeNativeHost.nativeFormula B graphs (rt.plan a).selected.config x entry i.val)
      rt (WholeNativeHost.execute B graphs rt) ∧
    ∀ a,rt.allocate=some a → rt.launchReady a=Bool.true → IndexCorrespondence a := by
  exact ⟨native_outcomes B graphs rt,fun a allocated ready => index_correspondence a (finite a allocated ready)⟩


#axiomsClean mv_native_fp32_source_outcomes

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholePublic
