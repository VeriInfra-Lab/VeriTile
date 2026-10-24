import bench.tritonbench_g.int8_matmul_kernel.Whole001.Kernels
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- Fixed-arity numeric leaves lift through evolving memory, the full grouped
grid, every legal completed autotune trial, source host behavior, fresh output,
and caller frame. This conclusion covers defined FP32 conversions and retains
all signed32 overflow; it does not equate native output to the unbounded sum. -/
specification whole_finite (r : Request) (caller : Set Nat) (before : Memory Word)
    (h : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15)
    (domain : WholeDomain r h.buffers before trials selected)
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (leaves : NumericLeaves cast instruction) :
    hostDecision r=.allocateAndLaunch ∧ ReturnContract r caller h.returned ∧
    (∀ m n,m<r.shape.m → n<r.shape.n →
      h.finalMemory (nativeArithmetic cast instruction) trials selected (outAddress r.shape h.buffers m n)=
        finiteTarget r h.output selected before m n) ∧
    (∀ p,p.1.1∈caller → h.finalMemory (nativeArithmetic cast instruction) trials selected p=before p) := by
  refine ⟨host_success r domain.host,returned_contract h,?_,?_⟩
  · intro m n hm hn
    rw [h.output_values _ trials selected domain.selectedLegal domain.trialsLegal m n hm hn]
    change callerAccumulation (nativeArithmetic cast instruction) r.shape h.buffers selected before m n=_
    rw [native_accumulation cast instruction leaves _ _ _ domain.selectedLegal before domain.defined m n hm]
    unfold callerAccumulation
    rw [software_reference _ _ domain.selectedLegal]
    rfl
  · intro p hp
    exact h.caller_frame _ trials selected domain.selectedLegal domain.trialsLegal p hp

/-- Whole native event composition. The remaining binding is the ordinary
single-event typed observation ABI and the two numerical leaf families above.
Physical CUDA loading, compiler correctness, launch success, and resources
are external execution bindings, never assumed as a whole-output equality. -/
specification whole_native (r : Request) (caller : Set Nat) (before : Memory Word)
    (h : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15)
    (domain : WholeDomain r h.buffers before trials selected)
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (leaves : NumericLeaves cast instruction)
    (abi : LocalEventABI Word Word σ (nativeArithmetic cast instruction))
    (state : σ) (entry : abi.observe state=h.initial) :
    (∀ m n,m<r.shape.m → n<r.shape.n →
      abi.observe ((h.launchCode trials selected).native abi state).2 (outAddress r.shape h.buffers m n)=
        finiteTarget r h.output selected before m n) ∧
    (∀ p,p.1.1∈caller → abi.observe ((h.launchCode trials selected).native abi state).2 p=before p) := by
  rw [h.native_launch _ trials selected domain.selectedLegal domain.trialsLegal abi state entry]
  exact (whole_finite r caller before h trials selected domain cast instruction leaves).2.2

#axiomsClean whole_finite
#axiomsClean whole_native
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
