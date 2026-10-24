import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real

/-! Original-source defect: the wrapper accepts contiguous lazy-negative A or
B views but passes their raw pointers without resolving view flags. The ABI
fact (data_ptr sees unchanged storage) is witnessed separately with Torch.
This theorem proves the resulting full ideal source execution disagrees with
the logical tensor product. It is not a native GPU experiment. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.LogicalViewDefect
open VeriTile.Triton FP.Structural
open Whole
noncomputable section

def request (c : Config) : Request where
  A := "A"; B := "B"; C := "C"
  rows := 1; inner := 1; innerB := 1; columns := 1
  strideAM := 1; strideAK := 1; strideBK := 1; strideBN := 1
  aContiguous := Bool.true; activation := ""
  target := if c=.hip128 then "hip" else "cuda"
  config := c

def rawState : State ℝ where
  mem := fun r _ => if r=("A" : RegionName) ∨ r=("B" : RegionName) then Cell.mk .fp32 1 else Cell.mk .fp16 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def rawA (c : Config) : Fin (request c).rows → Fin (request c).inner → ℝ := fun _ _ => 1
def rawB (c : Config) : Fin (request c).inner → Fin (request c).columns → ℝ := fun _ _ => 1
/-- torch._neg_view(A) leaves raw storage +1 but logical A is -1. -/
def logicalA (c : Config) : Fin (request c).rows → Fin (request c).inner → ℝ := fun _ _ => -1

theorem original_wrong_logical_result (c : Config) :
    ∃ final, host (idealAlgebra id) idealDot (request c) rawState=.ok ((request c).C,final) ∧
      final.mem "C" 0=Cell.mk .fp16 (1 : ℝ) ∧
      mathematical (request c) (logicalA c) (rawB c) ⟨0,by simp [request]⟩ ⟨0,by simp [request]⟩=(-1 : ℝ) ∧
      (1 : ℝ) ≠ mathematical (request c) (logicalA c) (rawB c) ⟨0,by simp [request]⟩ ⟨0,by simp [request]⟩ := by
  have ha : ∀ i k,inputA (request c) rawState i.val k.val=rawA c i k := by
    intro i k
    simp [inputA,request,rawState,rawA,Cell.read]
  have hb : ∀ k j,inputB (request c) rawState k.val j.val=rawB c k j := by
    intro k j
    simp [inputB,request,rawState,rawB,Cell.read]
  obtain ⟨final,he,hv,hf⟩ := host_correct (idealAlgebra id) idealDot (idealNumerics id)
    (ideal_primitives id) (request c) rawState (by simp [request]) (by simp [request]) rfl rfl
  have hout := hv 0 (by simp [request]) 0 (by simp [request])
  have hr := ideal_reference id (request c) rawState (rawA c) (rawB c) ha hb ⟨0,by simp [request]⟩ ⟨0,by simp [request]⟩
  have hone : reference (idealNumerics id) (request c) rawState 0 0=1 := by
    rw [hr]
    simp [mathematical,request,rawA,rawB]
  have hneg : mathematical (request c) (logicalA c) (rawB c) ⟨0,by simp [request]⟩ ⟨0,by simp [request]⟩=(-1 : ℝ) := by
    simp [mathematical,request,logicalA,rawB]
  refine ⟨final,he,?_,hneg,?_⟩
  · rw [hone] at hout
    simpa [outputOffset,request] using hout
  · rw [hneg]
    norm_num

#axiomsClean original_wrong_logical_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.LogicalViewDefect
