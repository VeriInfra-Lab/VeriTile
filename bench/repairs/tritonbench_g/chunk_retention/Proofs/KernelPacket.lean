import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHReal
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardOResult
import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardHReal
import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVResult
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHNative
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardONative
import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardHNative
import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVNativeResult

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelPacket
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000

inductive Stage where
  | forwardH | forwardO | backwardH | backwardQKV
  deriving DecidableEq
structure Call where
  shape : Shape
  stage : Stage
  scale : ℝ
  seed : Bool
  final : Bool
def Call.kernel (a : Call) : ComputeKernel := match a.stage with
  | .forwardH=>ForwardHSource.kernel a.shape a.seed a.final
  | .forwardO=>ForwardOSource.kernel a.shape a.scale
  | .backwardH=>BackwardHSource.kernel a.shape a.scale a.seed
  | .backwardQKV=>BackwardQKVSource.kernel a.shape a.scale

/-- Ordinary legal program coordinates and observed grid metadata only. -/
def Legal (a : Call) (s : State α) : Prop :=
  s.pids 2<a.shape.B*a.shape.H ∧ s.numPids 2=a.shape.B*a.shape.H ∧
  match a.stage with
  | .forwardH | .backwardH=>s.pids 0<a.shape.keyBlocks ∧ s.pids 1<a.shape.valueBlocks
  | .forwardO=>s.pids 0<a.shape.valueBlocks ∧ s.pids 1<a.shape.chunks
  | .backwardQKV=>s.pids 0<a.shape.keyBlocks ∧ s.pids 1<a.shape.chunks

inductive Observation (c : Shape) where
  | history (n : Nat) (i : ForwardHSource.Index c)
  | finalState (i : ForwardHSource.Index c)
  | output (i : ForwardOSource.Index c)
  | dhistory (n : Nat) (i : BackwardHSource.Index c)
  | query (i : BackwardQKVSource.KeyIndex c)
  | key (i : BackwardQKVSource.KeyIndex c)
  | valuePart (f : Nat) (i : BackwardQKVSource.ValueIndex c)

def Observation.region : Observation c→RegionName
  | .history _ _=>"h" | .finalState _=>"ht" | .output _=>"o"
  | .dhistory _ _=>"dh" | .query _=>"dq" | .key _=>"dk" | .valuePart _ _=>"dv"
def Observation.offset (s : State α) : Observation c→Nat
  | .history n i=>ForwardHSource.historyAddress c s n i
  | .finalState i=>ForwardHSource.stateAddress c s i
  | .output i=>ForwardOSource.outputAddress c s i
  | .dhistory n i=>BackwardHSource.historyAddress c s (BackwardHSource.chunk c n) i
  | .query i | .key i=>BackwardQKVSource.keyAddress c s i
  | .valuePart f i=>BackwardQKVSource.partialAddress c s f i
def Active (a : Call) (s : State α) : Observation a.shape→Prop
  | .history n i=>a.stage=.forwardH ∧ n<a.shape.chunks ∧ ForwardHSource.active a.shape s i
  | .finalState i=>a.stage=.forwardH ∧ a.final=Bool.true ∧ ForwardHSource.active a.shape s i
  | .output i=>a.stage=.forwardO ∧ ForwardOSource.active a.shape s i
  | .dhistory n i=>a.stage=.backwardH ∧ n<a.shape.chunks ∧ BackwardHSource.active a.shape s i
  | .query i | .key i=>a.stage=.backwardQKV ∧ BackwardQKVSource.keyActive a.shape s i
  | .valuePart f i=>a.stage=.backwardQKV ∧ f<a.shape.valueBlocks ∧ BackwardQKVSource.valueActive a.shape s f i
def Footprint (a : Call) (s : State α) (r : RegionName) (off : Nat) : Prop :=
  ∃ o, Active a s o ∧ r=o.region ∧ off=o.offset s
def Frame (a : Call) (s t : State α) : Prop :=
  ∀ r off, ¬Footprint a s r off → t.mem r off=s.mem r off

def RawFrame (a : Call) (s t : State α) : Prop := match a.stage with
  | .forwardH=>∀ (r : RegionName) (off : Nat),
      (r≠"h" ∨ ∀ n,n<a.shape.chunks → ∀ i,ForwardHSource.active a.shape s i → off≠ForwardHSource.historyAddress a.shape s n i) →
      (a.final=Bool.false ∨ r≠"ht" ∨ ∀ i,ForwardHSource.active a.shape s i → off≠ForwardHSource.stateAddress a.shape s i) → t.mem r off=s.mem r off
  | .forwardO=>∀ (r : RegionName) (off : Nat),
      (r≠"o" ∨ ∀ i,ForwardOSource.active a.shape s i → off≠ForwardOSource.outputAddress a.shape s i) → t.mem r off=s.mem r off
  | .backwardH=>∀ (r : RegionName) (off : Nat),
      (r≠"dh" ∨ ∀ n,n<a.shape.chunks → ∀ i,BackwardHSource.active a.shape s i → off≠BackwardHSource.historyAddress a.shape s n i) → t.mem r off=s.mem r off
  | .backwardQKV=>∀ (r : RegionName) (off : Nat),
      (r≠"dq" ∨ ∀ i,BackwardQKVSource.keyActive a.shape s i → off≠BackwardQKVSource.keyAddress a.shape s i) →
      (r≠"dk" ∨ ∀ i,BackwardQKVSource.keyActive a.shape s i → off≠BackwardQKVSource.keyAddress a.shape s i) →
      (r≠"dv" ∨ ∀ f,f<a.shape.valueBlocks → ∀ i,BackwardQKVSource.valueActive a.shape s f i → off≠BackwardQKVSource.partialAddress a.shape s f i) → t.mem r off=s.mem r off

theorem frame_of_raw (a : Call) (s t : State α) (h : RawFrame a s t) : Frame a s t := by
  classical
  intro r off safe
  cases hs : a.stage with
  | forwardH=>
    simp only [RawFrame,hs] at h
    apply h r off
    · by_cases hr : r="h"
      · right; intro n hn i hi he
        exact safe ⟨.history n i,⟨hs,hn,hi⟩,hr,he⟩
      · exact Or.inl hr
    · cases hf : a.final with
      | «false»=>exact Or.inl rfl
      | «true»=>
        right
        by_cases hr : r="ht"
        · right; intro i hi he
          exact safe ⟨.finalState i,⟨hs,hf,hi⟩,hr,he⟩
        · exact Or.inl hr
  | forwardO=>
    simp only [RawFrame,hs] at h
    apply h r off
    by_cases hr : r="o"
    · right; intro i hi he
      exact safe ⟨.output i,⟨hs,hi⟩,hr,he⟩
    · exact Or.inl hr
  | backwardH=>
    simp only [RawFrame,hs] at h
    apply h r off
    by_cases hr : r="dh"
    · right; intro n hn i hi he
      have hn' : a.shape.chunks-1-n<a.shape.chunks := by omega
      have he' : BackwardHSource.chunk a.shape (a.shape.chunks-1-n)=n := by
        dsimp [BackwardHSource.chunk]; omega
      apply safe
      refine ⟨.dhistory (a.shape.chunks-1-n) i,⟨hs,hn',hi⟩,hr,?_⟩
      simpa only [Observation.offset,he'] using he
    · exact Or.inl hr
  | backwardQKV=>
    simp only [RawFrame,hs] at h
    apply h r off
    · by_cases hr : r="dq"
      · right; intro i hi he
        exact safe ⟨.query i,⟨hs,hi⟩,hr,he⟩
      · exact Or.inl hr
    · by_cases hr : r="dk"
      · right; intro i hi he
        exact safe ⟨.key i,⟨hs,hi⟩,hr,he⟩
      · exact Or.inl hr
    · by_cases hr : r="dv"
      · right; intro f hf i hi he
        exact safe ⟨.valuePart f i,⟨hs,hf,hi⟩,hr,he⟩
      · exact Or.inl hr

/-- Independent Real retention/VJP targets; each intermediate history is an
ordinary input of the corresponding standalone kernel. -/
def realTarget (a : Call) (s : State ℝ) : Observation a.shape→ℝ
  | .history n i=>MathTarget.state a.shape (KernelReal.H.xs a.shape a.seed s) Bool.true (s.pids 2) (n*64)
      (ForwardHSource.feature a.shape s i.1.val) (ForwardHSource.column a.shape s i.2.1.val)
  | .finalState i=>MathTarget.finalState a.shape (KernelReal.H.xs a.shape a.seed s) Bool.true (s.pids 2)
      (ForwardHSource.feature a.shape s i.1.val) (ForwardHSource.column a.shape s i.2.1.val)
  | .output i=>KernelReal.O.target a.shape s a.scale i
  | .dhistory n i=>MathTarget.future a.shape (KernelReal.BH.xs a.shape a.seed s) a.scale (s.pids 2)
      (a.shape.boundary (a.shape.chunks-n))
      (BackwardHSource.feature a.shape s i.1.val) (BackwardHSource.column a.shape s i.2.1.val)
  | .query i=>KernelReal.QKV.queryTarget a.shape s a.scale i
  | .key i=>KernelReal.QKV.keyTarget a.shape s a.scale i
  | .valuePart f i=>KernelReal.QKV.partialTarget a.shape s a.scale f i
def RealResult (a : Call) (s : State ℝ) (read : RegionName→Nat→ℝ) : Prop :=
  ∀ o,Active a s o → read o.region (o.offset s)=realTarget a s o

theorem real_complete (a : Call) (s : State ℝ) (legal : Legal a s) :
    ∃ t,FP.Structural.exec (engine (withDot KernelReal.algebra KernelReal.dot)) a.kernel s=some t ∧
      RealResult a s (fun r off=>(t.mem r off).read .real) ∧ Frame a s t := by
  cases hs : a.stage with
  | forwardH=>
    obtain ⟨t,ht,hh,hf,fr⟩:=KernelReal.H.source_result a.shape a.seed a.final s
    refine ⟨t,by simpa only [Call.kernel,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      · exact hh _ _ ho.1 ho.2
      · exact hf ho.1 _ ho.2
      all_goals contradiction
    · simpa only [RawFrame,hs] using fr
  | forwardO=>
    obtain ⟨t,ht,hv,fr⟩:=KernelReal.O.source_result a.shape s a.scale
    refine ⟨t,by simpa only [Call.kernel,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      all_goals first | contradiction | exact hv _ ho
    · simpa only [RawFrame,hs] using fr
  | backwardH=>
    obtain ⟨t,ht,hv,fr⟩:=KernelReal.BH.source_result a.shape a.seed s a.scale
    refine ⟨t,by simpa only [Call.kernel,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      all_goals first | contradiction | exact hv _ _ ho.1 ho.2
    · simpa only [RawFrame,hs] using fr
  | backwardQKV=>
    have hc : s.pids 1<a.shape.chunks := by
      have hlegal := legal
      simp only [Legal,hs] at hlegal
      exact hlegal.2.2.2
    obtain ⟨t,ht,hq,hk,hv,fr⟩:=KernelReal.QKV.source_result a.shape s a.scale hc
    refine ⟨t,by simpa only [Call.kernel,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      · exact hq _ ho
      · exact hk _ ho
      · exact hv _ ho.1 _ ho.2
    · simpa only [RawFrame,hs] using fr

#axiomsClean frame_of_raw
#axiomsClean real_complete
#specNonCircular realTarget avoiding [run,FP.Structural.exec,ForwardHSource.update,BackwardQKVSource.queryValue]
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelPacket
