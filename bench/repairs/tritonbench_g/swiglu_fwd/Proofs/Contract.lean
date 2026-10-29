import bench.repairs.tritonbench_g.swiglu_fwd.Contract

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole

theorem checkPayload_iff (x : VeriTile.Torch.ArgumentMetadata) :
    checkPayload x = Bool.true ↔ ArgumentsAdmitted x := by
  cases x with
  | nonTensor => simp [checkPayload, ArgumentsAdmitted]
  | tensor x =>
    have fields : PayloadContract x ↔ x.layout = .strided ∧ x.dtype = .floating .fp32 ∧
        x.device.kind = "cuda" ∧ x.isNeg = Bool.false :=
      ⟨fun h => ⟨h.layout,h.dtype,h.cuda,h.noNeg⟩,fun ⟨a,b,c,d⟩ => ⟨a,b,c,d⟩⟩
    simp [checkPayload,ArgumentsAdmitted,fields,VeriTile.Torch.TensorMetadata.isCuda,and_assoc]

theorem checkLaunch_iff (m : LaunchMetadata) :
    checkLaunch m = Bool.true ↔ LaunchContract m := by
  have fields : LaunchContract m ↔
    0 < m.cols ∧ m.outputDevice = m.inputDevice ∧ m.rows < 2^31 ∧
    m.inputStride < 2^31 ∧ m.outputStride < 2^31 ∧
    m.rows*m.inputStride + m.cols+1024 < 2^31 ∧
    m.rows*m.outputStride + m.cols+1024 < 2^31 ∧
    (m.rows ≤ 1 ∨ m.cols ≤ m.outputStride) ∧
    (0 < m.rows → m.inputPointer % 4 = 0 ∧ m.outputPointer % 4 = 0) ∧
    (0 < m.rows →
      outputEnd m ≤ m.inputPointer ∨ inputEnd m ≤ m.outputPointer) :=
    ⟨fun h => ⟨h.positive,h.sameDevice,h.rows32,h.inputStride32,h.outputStride32,
        h.inputPadded32,h.outputPadded32,h.unique,h.alignment,h.separate⟩,
      fun ⟨a,b,c,d,e,f,g,h,i,j⟩ => ⟨a,b,c,d,e,f,g,h,i,j⟩⟩
  simp only [checkLaunch, Bool.and_eq_true, decide_eq_true_eq, fields, and_assoc]

#axiomsClean checkPayload_iff
#axiomsClean checkLaunch_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SwigluFwd
