import bench.tritonbench_g.sgmv_expand_slice.Proofs.IO
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridProgram
import VeriTile.Triton.Float.ReadProgramIO

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem schedule_run [Inhabited α] (M : Algebra α) (abi : SourceABI) (p : KernelArgs)
    (js : List Job) (s : State α) :
    runSchedule (js.map (invocation M abi p)) s = runSourceJobs M abi p js s := by
  induction js generalizing s with
  | nil => rfl
  | cons j js ih =>
    simpa only [List.map_cons,runSchedule,Invocation.run,Invocation.initial,invocation,
      bindRegisters,List.foldl_nil,Bool.false_eq_true,ite_false,ite_true,
      Option.getD_some,runSourceJobs] using
      congrArg (fun f => (FP.Structural.exec M (kernel abi {p with batch := j.batch})
        {s with
          pids := fun ax => if ax=0 then j.pid else 0
          numPids := fun ax => if ax=0 then j.count else 1}).bind f) (funext ih)

theorem binding_inputs [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (hi : abi.input=.single) (hw : abi.weight=.half) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (xs : ChannelValues α (inputChannels p shape segments))
    (s : State α) (binding : (io M abi p shape segments).InputBinding xs s)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) :
    SourceMemory.Inputs abi p shape s (inputValues xs) (weightValues xs) := by
  refine ⟨inner,columns,?_,?_⟩
  · intro r k hr hk
    have got := binding ⟨0,by simp [io]⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    change (s.mem p.input (r*p.inputRowStride+k*p.inputInnerStride)).read .fp32 =
      xs ⟨0,by simp⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) at got
    rw [hi]
    simpa [inputValues,matrixAt,hr,hk,ScalarFormat.dtype,toFloat] using got
  · intro a c k ha hc hk
    have got := binding ⟨1,by simp [io]⟩ (⟨a,ha⟩,⟨c,hc⟩,⟨k,hk⟩,PUnit.unit) trivial
    change (s.mem p.weight (p.adapterStride*a+k*p.weightInnerStride+c*p.weightColumnStride)).read .fp16 =
      xs ⟨1,by simp⟩ (⟨a,ha⟩,⟨c,hc⟩,⟨k,hk⟩,PUnit.unit) at got
    rw [hw]
    simpa [weightValues,matrixAt,ha,hc,hk,LowFormat.scalar,ScalarFormat.dtype,toFloat] using got

theorem binding_output [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (ho : abi.output=.single) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (xs : ChannelValues α (inputChannels p shape segments))
    (s : State α) (binding : (io M abi p shape segments).InputBinding xs s) :
    SourceSuffix.OutputMemory abi p shape s (entryValues xs) := by
  intro r c hr hc
  have got := binding ⟨2,by simp [io]⟩ (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
  change (s.mem p.output (r*p.outputRowStride+c*p.outputColumnStride)).read .real =
    xs ⟨2,by simp⟩ (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) at got
  unfold SourceSuffix.readOutput
  rw [ho]
  simpa [entryValues,matrixAt,hr,hc,SourceSuffix.storageType,toFloat] using got

#axiomsClean schedule_run
#axiomsClean binding_inputs
#axiomsClean binding_output
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
