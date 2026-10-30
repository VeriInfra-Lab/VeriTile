import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedStoreGeometry

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

inductive Mode where
  | forward | backward
  deriving DecidableEq

def kernel (mode : Mode) (s : Shape) (b : InputBuffers) (seed final : Bool) : ComputeKernel :=
  match mode with
  | .forward => Program.forward (layout s b) forwardRegions seed final
  | .backward => Program.backward (layout s b) backwardRegions seed final

def programStores [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) :=
  match mode with
  | .forward => forwardStores a s b plans initial seed final pid
  | .backward => backwardStores a s b plans initial seed final pid

def prepare (a : Arithmetic α) (s : Shape) (pid : ProgramId) (u : State (Staged.Node α)) :
    State (Staged.Node α) :=
  {u with regs := fun _ _ _ => none,pids := fun i => if i=0 then pid.iv else if i=1 then pid.ik else pid.head}
    |>.setReg "scale" .real [] (fun _ => Staged.tagged .scale (a.scale s.DK))

@[simp] theorem prepare_mem (a : Arithmetic α) (s : Shape) (pid : ProgramId) (u : State (Staged.Node α)) :
    (prepare a s pid u).mem=u.mem := rfl

@[simp] theorem prepare_launch (a : Arithmetic α) (s : Shape) (pid : ProgramId) (u : State (Staged.Node α)) :
    launch (prepare a s pid u)=pid := by cases pid; rfl

@[simp] theorem prepare_inputs [Inhabited α] (a : Arithmetic α) (s : Shape) (p : Layout)
    (pid : ProgramId) (u : State (Staged.Node α)) : inputs p (prepare a s pid u)=inputs p u := rfl

theorem programStores_congr [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (u v : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (he : inputs (layout s b) u=inputs (layout s b) v) :
    programStores mode a s b plans u seed final pid=programStores mode a s b plans v seed final pid := by
  have ho : outputWrites a s b plans u seed pid=outputWrites a s b plans v seed pid := by
    funext t; simp only [outputWrites,he]
  have hq : queryWrites a s b plans u seed pid=queryWrites a s b plans v seed pid := by
    funext t; simp only [queryWrites,he]
  have hr : reverseWrites a s b plans u final pid=reverseWrites a s b plans v final pid := by
    funext t; simp only [reverseWrites,reverseKeyWrites,reverseValueWrites,he]
  have hf : finalWrites a s b u seed final pid=finalWrites a s b v seed final pid := by
    simp only [finalWrites,he]
  cases mode <;> simp only [programStores,forwardStores,backwardStores,ho,hq,hr,hf]

theorem programStores_region [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (u : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (w : Write (Staged.Node α)) (hw : w∈programStores mode a s b plans u seed final pid) :
    outputRegion w.region := by
  cases mode
  · exact forwardStores_region _ _ _ _ _ _ _ _ w hw
  · exact backwardStores_region _ _ _ _ _ _ _ _ w hw

def gridPrograms (s : Shape) : List ProgramId :=
  (List.range s.NV).flatMap (fun iv => (List.range s.NK).flatMap (fun ik =>
    (List.range s.BH).map (fun h => ⟨iv,ik,h⟩)))

@[simp] theorem mem_gridPrograms (s : Shape) (pid : ProgramId) : pid∈gridPrograms s ↔ pid.Valid s := by
  cases pid
  simp [gridPrograms,ProgramId.Valid]

/-- Sequential grid composition of the actual shared quotation executor.
Every launch receives fresh scalar registers and the current memory. -/
def gridExec [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (seed final : Bool) : List ProgramId → State (Staged.Node α) → Option (State (Staged.Node α))
  | [],u => some u
  | pid::rest,u => (FP.Structural.exec (Staged.algebra a plans) (kernel mode s b seed final)
      (prepare a s pid u)).bind (gridExec mode a s b plans seed final rest)

def gridStores [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed final : Bool) (pids : List ProgramId) :=
  pids.flatMap (programStores mode a s b plans initial seed final)

theorem launch_exec_stores [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (p : PlanBinding s) (seed final : Bool) (u : State (Staged.Node α)) (pid : ProgramId)
    (shape : s.Admitted) :
    ∃ result,FP.Structural.exec (Staged.algebra a p.plans) (kernel mode s b seed final)
      (prepare a s pid u)=some result ∧
      result.mem=(writes (programStores mode a s b p.plans u seed final pid) u).mem := by
  have hs : (prepare a s pid u).regs .real [] "scale"=
      some (fun _ => Staged.tagged .scale (a.scale s.DK)) := by simp [prepare]
  have he := programStores_congr mode a s b p.plans (prepare a s pid u) u seed final pid rfl
  cases mode
  · obtain ⟨v,hv,hm⟩ := forward_full_stores a s b p.plans seed final (prepare a s pid u)
      (block_positive s shape).1 p.output.covers hs
    rw [prepare_launch] at hm
    have hm' := hm.trans (writes_mem_congr _ _ u rfl)
    change v.mem=(writes (programStores .forward a s b p.plans (prepare a s pid u) seed final pid) u).mem at hm'
    rw [he] at hm'
    exact ⟨v,hv,hm'⟩
  · obtain ⟨v,hv,hm⟩ := backward_full_stores a s b p.plans seed final (prepare a s pid u)
      (block_positive s shape).1 (block_positive s shape).2 p.dq.covers p.dk.covers p.dv.covers hs
    rw [prepare_launch] at hm
    have hm' := hm.trans (writes_mem_congr _ _ u rfl)
    change v.mem=(writes (programStores .backward a s b p.plans (prepare a s pid u) seed final pid) u).mem at hm'
    rw [he] at hm'
    exact ⟨v,hv,hm'⟩

theorem grid_exec_stores [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (p : PlanBinding s) (seed final : Bool) (initial u : State (Staged.Node α))
    (pids : List ProgramId) (shape : s.Admitted)
    (hin : inputs (layout s b) u=inputs (layout s b) initial) :
    ∃ result,gridExec mode a s b p.plans seed final pids u=some result ∧
      result.mem=(writes (gridStores mode a s b p.plans initial seed final pids) u).mem := by
  induction pids generalizing u with
  | nil => exact ⟨u,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨v,hv,hm⟩ := launch_exec_stores mode a s b p seed final u pid shape
    have hstores := programStores_congr mode a s b p.plans u initial seed final pid hin
    rw [hstores] at hm
    have hvin : inputs (layout s b) v=inputs (layout s b) initial := by
      have hm' : inputs (layout s b) v=inputs (layout s b)
          (writes (programStores mode a s b p.plans initial seed final pid) u) :=
        inputs_congr _ _ _ (fun r _ => congrFun hm r)
      rw [hm',writes_inputs _ _ _ (programStores_region mode a s b p.plans initial seed final pid),hin]
    obtain ⟨result,hr,hrm⟩ := ih v hvin
    refine ⟨result,?_,?_⟩
    · simp only [gridExec,hv,Option.bind_some,hr]
    · rw [gridStores,List.flatMap_cons,writes_append]
      exact hrm.trans (writes_mem_congr _ _ _ hm)

#axiomsClean launch_exec_stores
#axiomsClean grid_exec_stores
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
