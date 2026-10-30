import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOData

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section

theorem initial_eq {α : Type} (source : ComputeKernel) (model : Algebra α)
    (pid : Nat) (s : State α) : (launch source model pid).initial s=atProgram s pid := by
  simp [launch,Invocation.initial,bindRegisters,atProgram]

theorem schedule_eq {α : Type} [Inhabited α] (source : ComputeKernel) (model : Algebra α)
    (pids : List Nat) (s : State α) :
    runSchedule (pids.map (launch source model)) s=
      GridExecution.sequence (fun pid t=>FP.Structural.exec model source (atProgram t pid)) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,initial_eq,GridExecution.sequence,ih]
    rfl

theorem binding_preserved {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (jobs : List (Invocation α)) (x : ChannelValues α (IOData.inputs c))
    (initial current : State α) (binding : (IOData.program α c jobs).InputBinding x initial)
    (outside : Outside "C" initial current) :
    (IOData.program α c jobs).InputBinding x current := by
  intro port j active
  fin_cases port
  all_goals
    change (current.mem _ _).read _=_
    rw [outside _ _ (by simp [IOData.program,IOData.memory,IOData.inputs])]
    exact binding _ _ active

theorem output_binding {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (jobs : List (Invocation α)) (x : ChannelValues α (IOData.inputs c))
    (f : Nat → Nat → α) (i : Fin 17) (initial final : State α)
    (post : GridExecution.Post "C" (Geometry.address (Grid.geometry (params c.call i)))
      (fun j : Grid.Index (params c.call i)=>Cell.mk .fp16 (f j.1.val j.2.val))
      (fun _=>True) initial final) :
    (IOData.program α c jobs).OutputBinding x (IOData.expected c f) final ∧
      (IOData.program α c jobs).Frame x initial final := by
  constructor
  · intro port j active
    fin_cases port
    have hv := post.1 (j.1,j.2.1) trivial
    change mapCell id (final.mem "C" (c.C.addr j.1.val j.2.1.val))=Cell.mk .fp16 (f j.1.val j.2.1.val)
    change final.mem "C" (c.C.addr j.1.val j.2.1.val)=_ at hv
    rw [hv]
    rfl
  · intro r offset outside
    apply post.2 r offset
    by_cases hr : r="C"
    · right
      intro j _
      have ho := outside.1 (⟨0,by decide⟩ : Fin 1) (j.1,j.2,()) trivial
      change r≠"C" ∨ offset≠c.C.addr j.1.val j.2.val at ho
      exact ho.resolve_left (fun ne=>ne hr)
    · exact Or.inl hr

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOExecution
