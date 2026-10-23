import bench.repairs.tritonbench_g.layer_norm_liger.Correct
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.NativeCorrect
import bench.repairs.tritonbench_g.layer_norm_liger.FPEquiv
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelReal
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelFP
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelStrides

namespace VeriTile.Bench.Repairs.LayerNormLiger.TensorBridge
open VeriTile.Triton FP.Structural KernelContract
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
noncomputable section

/-- Successful complete forward body: population mean/rstd and every Y lane
are functions of actual Tensor logical values, with a physical-memory frame. -/
theorem layer_norm_forward_kernel_real (rest : Algebra ℝ) (c : ForwardCall)
    (pointers : RegionName → Nat) (memory : Nat → ℝ) (h : ForwardDomain c pointers) :
    NativeIndex.Safe c.rows c.columns 1 c.block ∧ ForwardAddresses c pointers ∧
    ∃ t, exec (withSum (Logical.algebra rest) Logical.sumPrimitive) (Forward.kernel c.parameters)
        (entry pointers memory c.pid) = some t ∧
      ForwardReal c memory (commit pointers c.writes memory t) ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers memory c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) → commit pointers c.writes memory t a = memory a) :=
  ⟨forward_native c h,forward_addresses c h,forward_real rest c h memory⟩

/-- Successful complete backward body, including empty groups, all DX cells,
both partial-gradient rows and the untouched physical-memory frame. -/
theorem layer_norm_backward_kernel_real (rest : Algebra ℝ) (c : BackwardCall)
    (pointers : RegionName → Nat) (memory : Nat → ℝ) (h : BackwardDomain c pointers) :
    NativeIndex.Safe c.rows c.columns c.programs c.block ∧ BackwardAddresses c pointers ∧
    ∃ t, exec (withSum (Logical.algebra rest) Logical.sumPrimitive) (Backward.kernel c.parameters)
        (entry pointers memory c.pid) = some t ∧
      BackwardReal c memory (commit pointers c.writes memory t) ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers memory c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) → commit pointers c.writes memory t a = memory a) ∧
      (∀ b ∈ c.inputs, ∀ i, i < b.size → b.logical (commit pointers c.writes memory t) i = b.logical memory i) := by
  obtain ⟨t,ran,values,frame,physical⟩ := backward_real rest c h memory
  exact ⟨backward_native c h,backward_addresses c h,t,ran,values,frame,physical,backward_input_frame c h memory t⟩

/-- FP32 source semantics with an explicit compiler primitive/contraction/tree
binding. Every physical output cell equals the independent ordered graph over
actual Tensor logical values. No numerical experiment premise is used. -/
theorem layer_norm_forward_kernel_fp32 [Inhabited (FP.Value .fp32)] (P : NativePlan.Primitives)
    (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (c : ForwardCall) (pointers : RegionName → Nat) (memory : Nat → FP.Value .fp32)
    (h : ForwardDomain c pointers) :
    NativeIndex.Safe c.rows c.columns 1 c.block ∧ ForwardAddresses c pointers ∧
    ∃ t, exec (withSum (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees))
        (Forward.kernel c.parameters) (entry pointers (words memory) c.pid) = some t ∧
      evaluateMemory P (commit pointers c.writes (words memory) t) = forwardGraph P contraction trees c pointers memory ∧
      ForwardNative c pointers (forwardGraph P contraction trees c pointers memory)
        (evaluateMemory P (commit pointers c.writes (words memory) t)) ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers (words memory) c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) →
        evaluateMemory P (commit pointers c.writes (words memory) t) a = memory a) :=
by
  obtain ⟨t,ran,values,frame,physical⟩ := forward_fp P contraction trees c h memory
  exact ⟨forward_native c h,forward_addresses c h,t,ran,values,forward_native_outputs c h _ _ values,frame,physical⟩

/-- The complete backward loop preserves the selected FP32 source order,
FMA choices, casts and reductions; the result includes DX and both partials. -/
theorem layer_norm_backward_kernel_fp32 [Inhabited (FP.Value .fp32)] (P : NativePlan.Primitives)
    (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (c : BackwardCall) (pointers : RegionName → Nat) (memory : Nat → FP.Value .fp32)
    (h : BackwardDomain c pointers) :
    NativeIndex.Safe c.rows c.columns c.programs c.block ∧ BackwardAddresses c pointers ∧
    ∃ t, exec (withSum (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees))
        (Backward.kernel c.parameters) (entry pointers (words memory) c.pid) = some t ∧
      evaluateMemory P (commit pointers c.writes (words memory) t) = backwardGraph P contraction trees c pointers memory ∧
      BackwardNative c pointers (backwardGraph P contraction trees c pointers memory)
        (evaluateMemory P (commit pointers c.writes (words memory) t)) ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers (words memory) c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) →
        evaluateMemory P (commit pointers c.writes (words memory) t) a = memory a) ∧
      (∀ b ∈ c.inputs, ∀ i, i < b.size →
        b.logical (evaluateMemory P (commit pointers c.writes (words memory) t)) i = b.logical memory i) := by
  obtain ⟨t,ran,values,frame,physical⟩ := backward_fp P contraction trees c h memory
  exact ⟨backward_native c h,backward_addresses c h,t,ran,values,backward_native_outputs c h _ _ values,
    frame,physical,backward_fp_input_frame P c h memory t⟩

#axiomsClean layer_norm_forward_kernel_real
#axiomsClean layer_norm_forward_kernel_fp32
#axiomsClean layer_norm_backward_kernel_real
#axiomsClean layer_norm_backward_kernel_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.TensorBridge
