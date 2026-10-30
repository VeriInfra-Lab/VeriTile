import bench.tritonbench_g.int8_matmul_kernel.Whole001.Kernels
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- Full mathematical source interpretation: original caller words, all
selected output cells, arbitrary completed legal autotune trials, fresh return
metadata, and preservation of every caller-owned typed memory observation. -/
specification whole_mathematical (r : Request) (caller : Set Nat) (before : Memory Word)
    (h : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15)
    (domain : WholeDomain r h.buffers before trials selected) :
    hostDecision r=.allocateAndLaunch ∧ ReturnContract r caller h.returned ∧
    (∀ m n,m<r.shape.m → n<r.shape.n →
      (h.math.finalMemory integerArithmetic trials selected (outAddress r.shape h.buffers m n)).2=
        quantizedDenseTarget r h.output before m n) ∧
    (∀ p,p.1.1∈caller → h.math.finalMemory integerArithmetic trials selected p=mathLift before p) := by
  refine ⟨host_success r domain.host,returned_contract h,?_,?_⟩
  · intro m n hm hn
    change (h.math.finalMemory integerArithmetic trials selected (outAddress r.shape h.math.buffers m n)).2=_
    rw [h.math.output_values integerArithmetic trials selected domain.selectedLegal domain.trialsLegal m n hm hn]
    change accumulation (mathematical (fun x : MathCell => referenceCast x.1)
      (fun x : MathCell => packedByte x.1) (fun z => ((0 : Word),z))) r.shape selected
      (fun k => mathLift before (aAddress h.buffers m k))
      (fun k => mathLift before (bAddress h.buffers k n)) = _
    rw [accumulation_integer _ _ _ _ _ domain.selectedLegal]
    rfl
  · intro p hp
    exact h.math.caller_frame integerArithmetic trials selected domain.selectedLegal domain.trialsLegal p hp

#axiomsClean whole_mathematical
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
