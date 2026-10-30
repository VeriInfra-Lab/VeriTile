import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOEvents
import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOByteQKV

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOJob
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2500000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def writers (launch : Fin order.length) : List (IO.Event order) :=
  (List.finRange 3).flatMap fun phase=>(TileShape.allIndices [16,64]).map fun lane=>⟨launch,phase,lane⟩

theorem binding_preserved (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (s t : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s)
    (same : ∀ phase,InputMemory (qkvArgument p phase) t s) :
    (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals t := by
  intro i off
  change Fin 9 at i
  have memory : t.mem (IO.inputs fp8 p)[i].buf=s.mem (IO.inputs fp8 p)[i].buf := by
    fin_cases i
    all_goals first
      | exact (same 0).1 | exact (same 1).1 | exact (same 2).1
      | exact (same 0).2.1 | exact (same 1).2.1 | exact (same 2).2.1
      | exact (same 0).2.2 | exact (same 1).2.2 | exact (same 2).2.2
  change (t.mem (IO.inputs fp8 p)[i].buf off).read (IO.inputs fp8 p)[i].dtype=_
  rw [memory]
  exact hin i off

theorem qkv_inputs (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (s : State α)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (reader : Fin 3) : InputMemory (qkvArgument p reader) (qkvWrite a native fp8 p s epsilon theta) s := by
  let q:=calleeWrite a native (selectedRotaryPlan 0) fp8 Bool.false (qkvArgument p 0) s s epsilon theta
  let k:=calleeWrite a native (selectedRotaryPlan 1) fp8 fp8 (qkvArgument p 1) s q epsilon theta
  exact InputMemory.trans _ _ k s
    (calleeWrite_inputs a native (selectedRotaryPlan 2) fp8 fp8 (qkvArgument p 2)
      (qkvArgument p reader) s k epsilon theta (apart 2 reader))
    (InputMemory.trans _ k q s
      (calleeWrite_inputs a native (selectedRotaryPlan 1) fp8 fp8 (qkvArgument p 1)
        (qkvArgument p reader) s q epsilon theta (apart 1 reader))
      (calleeWrite_inputs a native (selectedRotaryPlan 0) fp8 Bool.false (qkvArgument p 0)
        (qkvArgument p reader) s s epsilon theta (apart 0 reader)))

theorem qkv_scatter (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (s : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s)
    (launch : Fin order.length) (pid : s.pids=order[launch]) :
    qkvWrite a native fp8 p s epsilon theta=
      (writers launch).foldl (IOEvents.put a native fp8 p epsilon theta vals) s := by
  have target (phase : Fin 3) :
      calleeTarget a native (selectedRotaryPlan phase) fp8 (qkvArgument p phase) s epsilon theta=
        fun lane=>Target.raw a native fp8 p epsilon theta vals ⟨launch,phase,lane⟩ := by
    funext lane
    exact IOBinding.target a native fp8 p epsilon theta order vals s hin ⟨launch,phase,lane⟩ pid
  simp only [writers,show List.finRange 3=[0,1,2] from rfl,List.flatMap_cons,List.flatMap_nil,
    List.append_nil,List.foldl_append,List.foldl_map]
  simp only [qkvWrite,calleeWrite,tileWrite,target,IOEvents.put,IOEvents.value,
    IO.eventActive,IO.eventRegion,IO.eventOffset,programRow,programCol,pid]
  rfl

theorem success (a : Arithmetic α) (zero : ZeroLaws a) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (domain : IO.InputDomain fp8 p vals)
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (s : State α) (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s)
    (launch : Fin order.length) :
    ∃ final,(IO.job a native fp8 p epsilon theta order[launch]).run s=some final ∧
      final.mem=((writers launch).foldl (IOEvents.put a native fp8 p epsilon theta vals) s).mem ∧
      (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals final := by
  let initial:=(IO.job a native fp8 p epsilon theta order[launch]).initial s
  have ib : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals initial := by
    intro i off
    simpa only [initial,Invocation.initial_mem] using hin i off
  have he : initial.regs .real [] "EPS"=some (fun _=>epsilon) := by
    simp [initial,IO.job,Invocation.initial,bindRegisters]
  have ht : initial.regs .real [] "THETA"=some (fun _=>theta) := by
    simp [initial,IO.job,Invocation.initial,bindRegisters]
  have pid : initial.pids=order[launch] := rfl
  obtain ⟨final,ran,memory,_,_,_⟩:=IOBytes.qkv_complete a zero native fp8 p initial epsilon theta he ht
    (IOBinding.bytes a native fp8 p epsilon theta order vals initial ib domain) even apart
  refine ⟨final,ran,?_,binding_preserved a native fp8 p epsilon theta order vals initial final ib ?_⟩
  · rw [memory,qkv_scatter a native fp8 p epsilon theta order vals initial ib launch pid]
    funext r off
    rw [IOEvents.scatter_at,IOEvents.scatter_at]
    rfl
  · intro reader
    simpa only [InputMemory,memory] using qkv_inputs a native fp8 p epsilon theta initial apart reader

#axiomsClean qkv_scatter
#axiomsClean success
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOJob
