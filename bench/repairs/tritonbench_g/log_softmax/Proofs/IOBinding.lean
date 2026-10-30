import bench.repairs.tritonbench_g.log_softmax.IO
import bench.repairs.tritonbench_g.log_softmax.Proofs.KernelSafety
import bench.repairs.tritonbench_g.log_softmax.Proofs.Observed

namespace VeriTile.Bench.Repairs.LogSoftmax.IOBinding
open VeriTile.Triton FP.Structural
noncomputable section

theorem launch_initial (direction : Direction) (algebra : Algebra β) (candidate : Bool)
    (M N K BM C : Nat) (p : Nat × Nat) (s : State β) :
    (IO.launch direction algebra candidate M N K BM C p).initial s=
      KernelGrid.positioned s p.1 p.2 := by
  simp [IO.launch,Invocation.initial,bindRegisters,KernelGrid.positioned]

theorem engine_run [Inhabited A] (direction : Direction) (engine : Kernel.Engine A C)
    (candidate : Bool) (M N K BM : Nat) (order : List (Nat × Nat)) (s : State A) :
    runSchedule (order.map (IO.launch direction engine.algebra candidate M N K BM C)) s=
      KernelGrid.run direction engine candidate M N K BM order s := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
      simp only [List.map_cons,runSchedule,Invocation.run,launch_initial]
      change (exec engine.algebra (KernelGrid.program direction candidate M N K BM C)
        (KernelGrid.positioned s p.1 p.2)).bind _ = _
      rw [KernelGrid.run]
      cases h : exec engine.algebra (KernelGrid.program direction candidate M N K BM C)
        (KernelGrid.positioned s p.1 p.2) <;> simp [h,ih]

theorem observed_run [Inhabited A] (direction : Direction) (algebra : Algebra A)
    (candidate : Bool) (M N K BM C : Nat) (order : List (Nat × Nat)) (s : State A) :
    runSchedule (order.map (IO.launch direction (FP.ProfiledRewrite.engine algebra)
      candidate M N K BM C)) s=Observed.run direction algebra candidate M N K BM C order s := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
      simp only [List.map_cons,runSchedule,Invocation.run,launch_initial]
      change (exec (FP.ProfiledRewrite.engine algebra)
        (KernelGrid.program direction candidate M N K BM C)
        (KernelGrid.positioned s p.1 p.2)).bind _ = _
      rw [Observed.run]
      cases h : exec (FP.ProfiledRewrite.engine algebra)
        (KernelGrid.program direction candidate M N K BM C)
        (KernelGrid.positioned s p.1 p.2) <;> simp [h,ih]

theorem schedule_safe [Inhabited β] (direction : Direction) (encode : α → β)
    (observe : β → α) (algebra : Algebra β) (candidate : Bool) (M N K BM C : Nat)
    (hc : 0<C) (bounds : RegionName → Nat)
    (vals : (IO.grid direction encode observe algebra candidate M N K BM C).Inputs)
    (bounded : (IO.grid direction encode observe algebra candidate M N K BM C).InBounds bounds vals)
    (order : List (Nat × Nat)) (s : State β) :
    ScheduleSafe bounds (order.map (IO.launch direction algebra candidate M N K BM C)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons p rest ih =>
      refine ⟨?_,fun _ _ => ih _⟩
      simp only [Invocation.Safe,launch_initial]
      cases direction
      · have input := bounded.2.1
        have output := bounded.2.2.1
        have safe := KernelSafety.forward_source algebra "output" "input" M N K BM C
          (KernelGrid.positioned s p.1 p.2) hc candidate bounds
          (fun i hi => input (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial)
          (fun i hi => output (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial)
        cases candidate <;> exact safe
      · have input := bounded.2.1
        have output := bounded.2.2.1
        have safe := KernelSafety.backward_source algebra "input" "gradient" "output"
          M N K BM C (KernelGrid.positioned s p.1 p.2) candidate bounds
          (fun i hi => input (0 : Fin 2) (⟨i,hi⟩,PUnit.unit) trivial)
          (fun i hi => input (1 : Fin 2) (⟨i,hi⟩,PUnit.unit) trivial)
          (fun i hi => output (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial)
        cases candidate <;> exact safe

theorem frame (direction : Direction) (encode : α → β) (observe : β → α)
    (algebra : Algebra β) (candidate : Bool) (M N K BM C : Nat)
    (vals : (IO.grid direction encode observe algebra candidate M N K BM C).Inputs)
    (initial final : State β)
    (preserved : ∀ (r : RegionName) i, r ≠ "output" ∨ M * N * K ≤ i →
      final.mem r i=initial.mem r i) :
    (IO.grid direction encode observe algebra candidate M N K BM C).Frame vals initial final := by
  intro r i outside
  apply preserved r i
  by_cases hr : r="output"
  · right
    by_contra hi
    have active : i<M*N*K := by omega
    cases direction <;>
      have impossible := outside.1 (0 : Fin 1) (⟨i,active⟩,PUnit.unit) trivial
    all_goals simpa [IO.grid,IO.forward,IO.backward,hr] using impossible
  · exact Or.inl hr

end
end VeriTile.Bench.Repairs.LogSoftmax.IOBinding
