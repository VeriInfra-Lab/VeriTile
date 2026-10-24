import bench.tritonbench_g.relu_triton_kernel.Proofs.FP64Host

/-! Public additive fp64 source behavior contracts. These prove the actual
pid-zero-only program, not whole-tensor mathematical ReLU when N > 1024.
Typed total memory is a primitive view; this module does not add float byte
machinery or claim repository-wide original-program acceptance. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Correct
open VeriTile.Triton GatedGrid FP64Source
noncomputable section

specification relu_fp64_source_grid_correctness
    (num : Numerics) (x y : RegionName) (N : Nat)
    (_distinct : x ≠ y) (domain : FiniteAddress.IndexDomain N)
    (order : List Nat) (complete : order.Perm (List.range (gridCount N 1024)))
    (s : State 1024) :
    (runSchedule num x y N 1024 order s).mem =
      (if gridCount N 1024 = 0 then s.mem else reference num x y N 1024 s.mem) ∧
    (scheduleTrace num x y N 1024 order s).map Prod.fst = order ∧
    (∀ pid before after,
      (pid, before, after) ∈ scheduleTrace num x y N 1024 order s →
      after = run num x y N 1024 pid before ∧
      after.regs = computedRegisters x N 1024 pid before) ∧
    (∀ r o, r ≠ y ∨ 1024 ≤ o ∨ N ≤ o →
      (runSchedule num x y N 1024 order s).mem.f32 r o = s.mem.f32 r o) ∧
    (runSchedule num x y N 1024 order s).mem.f64 = s.mem.f64 ∧
    (∀ pid, pid ∈ order → ∀ (before : State 1024) (i : Fin 1024),
      (FiniteAddress.offset32 pid i.val).toNat = (run num x y N 1024 pid before).regs.offsets i ∧
      FiniteAddress.mask32 N pid i.val = (run num x y N 1024 pid before).regs.mask i ∧
      FiniteAddress.stores32 pid = decide ((run num x y N 1024 pid before).regs.pid = 0)) := by
  refine ⟨grid_memory num x y N 1024 order s complete,
    trace_pids num x y N 1024 order s,
    fun pid before after entry => trace_executes_source num x y N 1024 order s pid before after entry,
    fun r o outside => grid_frame num x y r N 1024 o order s outside,
    grid_input_frame num x y N 1024 order s, ?_⟩
  intro pid hp before i
  have launched : pid < gridCount N 1024 := List.mem_range.mp (complete.mem_iff.mp hp)
  exact registers_finite_address num x y N pid before i domain launched

open HostWrapper

specification relu_fp64_host_source_correctness
    {α : Type} [Inhabited α] (P : Primitives (FP64Host.Runtime α))
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : Numerics) (undef : Nat → F64) (order : Nat → List Nat)
    (x output : Tensor) (s d a m : FP64Host.Runtime α)
    (trace : List Event) (device : String) (n : Nat)
    (dtype : x.dtype = "float64")
    (_outputDType : output.dtype = "float32") (_distinct : x.region ≠ output.region)
    (_domain : FiniteAddress.IndexDomain n)
    (complete : (order (gridCount n 1024)).Perm (List.range (gridCount n 1024)))
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    ∃ final, relu (FP64Host.withLogicalLaunch P M cmp16 cmp32 num undef order) x ⟨s, trace⟩ =
        .ok output ⟨final, trace ++ successTrace x output device n⟩ ∧
      final.2 = (if gridCount n 1024 = 0 then m.2 else
        reference num x.region output.region n 1024 m.2) ∧
      final.1 = m.1 :=
  FP64Host.relu_fp64_success P M cmp16 cmp32 num undef order x output s d a m trace device n
    dtype complete hd ha hm

#axiomsClean relu_fp64_source_grid_correctness
#axiomsClean relu_fp64_host_source_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Correct
