import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedPayloads
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedGridCoverage

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem grid_execution_memory [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (p : PlanBinding s) (initial : State (Staged.Node α)) (seed final : Bool) (domain : StoreDomain s b) :
    ∃ result,gridExec mode a s b p.plans seed final (gridPrograms s) initial=some result ∧
      ∀ r i,result.mem r i=if writtenRange mode s final r i then
        payload a s p.plans (inputs (layout s b) initial) seed final r i else initial.mem r i := by
  obtain ⟨result,he,hm⟩ := grid_exec_stores mode a s b p seed final initial initial (gridPrograms s) domain.shape rfl
  refine ⟨result,he,?_⟩
  intro r i
  rw [hm,writes_read (target := payload a s p.plans (inputs (layout s b) initial) seed final)]
  · cases mode
    · rw [forward_coverage a s b p.plans initial seed final domain]; rfl
    · rw [backward_coverage a s b p.plans initial seed final domain]; rfl
  · intro w hw
    rcases List.mem_flatMap.mp hw with ⟨pid,hpid,hw⟩
    have hp := (mem_gridPrograms s pid).mp hpid
    cases mode
    · exact forwardStores_payload a s b p.plans initial seed final pid domain hp w hw
    · exact backwardStores_payload a s b p.plans initial seed final pid domain hp w hw

/-- Launch memory is bound to the actual strided tensor storage. Equal storage
identities among inputs remain aliases; the six source pointers stay separate
names only inside Structural's typed-region representation. -/
def inputTensor (b : InputBuffers) (r : RegionName) : Tensor4 :=
  if r="q" then b.q else if r="k" then b.k else if r="v" then b.v else if r="do" then b.do_
  else if r="initial_state" then b.seed else b.finalGrad

def initialState [Inhabited α] (a : Arithmetic α) (b : InputBuffers) (m : Memory α) : State (Staged.Node α) where
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d
  mem := fun r i => .mk .real (Staged.atom (m ((inputTensor b r).storage,((inputTensor b r).offset+i : Nat))))

@[simp] theorem initialState_inputs [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (m : Memory α) :
    inputs (layout s b) (initialState a b m)=b.observe s m := by
  unfold inputs InputBuffers.observe readWord initialState inputTensor
  simp only [Region.mk.injEq,String.reduceEq,reduceIte,Cell.read_same,Staged.atom,layout]
  congr 1 <;> funext h t c <;>
    simp only [InputBuffers.qAddress,InputBuffers.kAddress,InputBuffers.vAddress,
      InputBuffers.doAddress,InputBuffers.seedAddress,InputBuffers.finalAddress,Tensor4.address,stateOffset]
  all_goals congr 2 <;> ring

def region : PartialKind → RegionName
  | .output => "o"
  | .dq => "dq"
  | .dk => "dk"
  | .dv => "dv"
  | .finalState => "final_state"

@[simp] theorem payload_read [Inhabited α] (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (kind : PartialKind) (i : Nat) :
    ((payload a s plans x seed final (region kind) i).read .real).value=sourceCell a s plans x seed final kind i := by
  cases kind <;> simp [payload,region,Cell.read,Staged.atom]

theorem grid_read [Inhabited α] (mode : Mode) (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (p : PlanBinding s) (seed final : Bool) (initial result : State (Staged.Node α))
    (hm : ∀ r i,result.mem r i=if writtenRange mode s final r i then
      payload a s p.plans (inputs (layout s b) initial) seed final r i else initial.mem r i)
    (kind : PartialKind) (i : Nat) (hi : writtenRange mode s final (region kind) i) :
    readWord result (region kind) i=sourceCell a s p.plans (inputs (layout s b) initial) seed final kind i := by
  unfold readWord
  rw [hm,if_pos hi,payload_read]

#axiomsClean grid_execution_memory
#axiomsClean initialState_inputs
#axiomsClean grid_read
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
