import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOSafetyQKV

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
variable (a : Arithmetic α) (zero : ZeroLaws a) (native fp8 : Bool)
variable (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
variable (vals : ArrayValues α (IO.inputs fp8 p)) (domain : IO.InputDomain fp8 p vals)
variable (even : ∀ phase,(p phase).N%2=0)
variable (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
  (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
variable (capacity : RegionName → Nat)
variable (bounds : (IO.qkvIO a native fp8 p epsilon theta order).InBounds capacity vals)
include zero domain even apart bounds

theorem job_safe (s : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s)
    (launch : Fin order.length) : (IO.job a native fp8 p epsilon theta order[launch]).Safe capacity s := by
  let initial:=(IO.job a native fp8 p epsilon theta order[launch]).initial s
  have ib : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals initial := by
    intro i off
    simpa only [initial,Invocation.initial_mem] using hin i off
  have he : initial.regs .real [] "EPS"=some (fun _=>epsilon) := by
    simp [initial,IO.job,Invocation.initial,bindRegisters]
  have ht : initial.regs .real [] "THETA"=some (fun _=>theta) := by
    simp [initial,IO.job,Invocation.initial,bindRegisters]
  have pid : initial.pids=order[launch] := rfl
  have bound (phase : Fin 3) : Bounds (qkvArgument p phase) initial capacity := by
    constructor
    · intro m k hk
      simpa only [programRow,programCol,pid,qkvArgument,columns] using
        IOCoverage.x_bound a native fp8 p epsilon theta order vals capacity bounds phase order[launch]
          (List.getElem_mem launch.isLt) m k hk
    · intro n k hk
      simpa only [programRow,programCol,pid,qkvArgument,columns] using
        IOCoverage.w_bound a native fp8 p epsilon theta order vals capacity bounds phase order[launch]
          (List.getElem_mem launch.isLt) n k hk
    · intro k hk
      exact IOCoverage.rms_bound a native fp8 p epsilon theta order vals capacity bounds phase order[launch]
        (List.getElem_mem launch.isLt) k hk
    · intro lane active
      exact IOCoverage.write_bound a native fp8 p epsilon theta order vals capacity bounds
        ⟨launch,phase,lane⟩ active
  exact qkv_safe a zero native fp8 p initial epsilon theta he ht
    (IOBinding.bytes a native fp8 p epsilon theta order vals initial ib domain) even apart capacity bound

theorem schedule_indices (launches : List (Fin order.length)) (s : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s) :
    ScheduleSafe capacity (launches.map fun i=>IO.job a native fp8 p epsilon theta order[i]) s := by
  induction launches generalizing s with
  | nil=>trivial
  | cons launch rest ih=>
    refine ⟨job_safe a zero native fp8 p epsilon theta order vals domain even apart capacity bounds s hin launch,?_⟩
    intro next ran
    obtain ⟨final,success,_,binding⟩:=IOJob.success a zero native fp8 p epsilon theta order vals domain even apart s hin launch
    have equal : next=final := Option.some.inj (ran.symm.trans success)
    subst next
    exact ih final binding

theorem schedule_safe (s : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s) :
    ScheduleSafe capacity ((IO.qkvIO a native fp8 p epsilon theta order).schedule vals) s := by
  have launchEq :
      (List.finRange order.length).map (fun i=>IO.job a native fp8 p epsilon theta order[i])=
        order.map (IO.job a native fp8 p epsilon theta) := by
    simpa only [List.map_map,Function.comp_def] using
      congrArg (List.map (IO.job a native fp8 p epsilon theta)) (List.map_getElem_finRange order)
  simpa only [IO.qkvIO,launchEq] using schedule_indices a zero native fp8 p epsilon theta order vals domain even apart
    capacity bounds (List.finRange order.length) s hin

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
