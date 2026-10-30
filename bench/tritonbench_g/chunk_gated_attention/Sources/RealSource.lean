import bench.tritonbench_g.chunk_gated_attention.Sources.DirectRealWhole
import bench.tritonbench_g.chunk_gated_attention.Sources.DirectCumulativeWhole
import bench.tritonbench_g.chunk_gated_attention.Sources.HostCoordinates

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RealSource
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open HostStorage
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

abbrev Memory := Nat→ℝ
def layout (r : Admission.Inner) := HostShape.recurrent r.T r.K r.V r.BT r.BK r.BV
def regions : RealHProgram.Regions := ⟨"k","v","g","h","h0","ht"⟩
def logical (before : Memory) (tensor : Admission.Tensor) (times features h t f : Nat) : ℝ :=
  before (tensor.byteBase+4*(h*(times*features)+t*features+f))
def inputs (r : Admission.Inner) (before : Memory) (h : Nat) : Host.InnerInputs where
  key:=logical before r.key r.T r.K h
  value:=logical before r.value r.T r.V h
  cumulativeGate:=logical before r.gate r.T (if r.gateK then r.K else r.V) h
  seed:=fun k v=>match r.seed with | none=>0 | some tensor=>logical before tensor r.K r.V h k v

def inputState (r : Admission.Inner) (before : Memory) (scratch : BlockState) : BlockState :=
  let a:=Materialize.inputs (layout r) r.T r.K r.V r.gateK (inputs r before)
  {scratch with mem:=fun region i=>
    if region="k" then .real (a.key i) else if region="v" then .real (a.value i)
    else if region="g" then .real (a.gate i) else if region="h0" then .real (a.seed i) else scratch.mem region i}

def History (r : Admission.Inner) (before : Memory) (i : Nat) : ℝ :=
  Host.innerState (inputs r before (i/(r.config.NT*r.K*r.V))) r.T r.BT r.gateK r.seed.isSome
    ((i/(r.K*r.V))%r.config.NT) ((i/r.V)%r.K) (i%r.V)
def Final (r : Admission.Inner) (before : Memory) (i : Nat) : ℝ :=
  Host.innerState (inputs r before (i/(r.K*r.V))) r.T r.BT r.gateK r.seed.isSome
    r.config.NT ((i/r.V)%r.K) (i%r.V)

theorem inner_exists (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState) :
    ∃ state,DirectRealSequence.Runs (layout r) regions .fp32 .fp32 .fp32 .fp32 .fp32
      r.gateK r.seed.isSome r.final.isSome Bool.false
      (LaunchPlan.plans (r.config.heads*(r.config.PK/r.BK)*(r.config.PV/r.BV)))
      {inputState r before scratch with undef:=fun _ _=>0} state ∧
      DirectRealWhole.Outputs (layout r) (r.B*r.H) r.T r.K r.V regions (inputs r before)
        r.gateK r.seed.isSome r.final.isSome state ∧
      BlockState.WriteWithin (DirectRealSequence.allWrites (layout r) regions
        (LaunchPlan.plans (r.config.heads*(r.config.PK/r.BK)*(r.config.PV/r.BV)))
        r.final.isSome (inputState r before scratch)) (inputState r before scratch) state := by
  have blocks:=r.blocks checked
  apply DirectRealWhole.executes (layout r) (r.B*r.H) r.T r.K r.V regions .fp32 .fp32 .fp32 .fp32 .fp32
    r.gateK r.seed.isSome r.final.isSome Bool.false (inputs r before)
    (HostShape.recurrent_canonical _ _ _ _ _ _ blocks.1 blocks.2.1 blocks.2.2)
    (HostShape.le_padded _ _ blocks.2.1) (HostShape.le_padded _ _ blocks.2.2)
  · constructor <;> simp [regions]
  · simp [regions]
  all_goals intro off; simp [inputState,regions,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of]

def innerResult (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState) : BlockState :=
  (inner_exists r checked before scratch).choose

theorem history (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState)
    (i : Nat) (hi : i < innerCount r) :
    (innerResult r checked before scratch).readMem "h" (HostData.historyOffset r i)=History r before i := by
  have hb:=HostCoordinates.decode4 hi
  have outputs:=(inner_exists r checked before scratch).choose_spec.2.1
  simpa [innerResult,regions,HostData.historyOffset,DirectRealWhole.historyOffset,History,
    layout,HostShape.recurrent,HostFP.Inner.NT,Admission.Inner.config,Nat.mul_assoc] using
    outputs.history _ _ _ _ hb.1 hb.2.1 hb.2.2.1 hb.2.2.2

theorem final (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState)
    (enabled : r.final.isSome=Bool.true) (i : Nat) (hi : i<finalCount r) :
    (innerResult r checked before scratch).readMem "ht" (HostData.finalOffset r i)=Final r before i := by
  have hb:=HostCoordinates.decode3 hi
  have outputs:=(inner_exists r checked before scratch).choose_spec.2.1
  simpa [innerResult,regions,HostData.finalOffset,DirectRealWhole.finalOffset,Final,
    layout,HostShape.recurrent,HostFP.Inner.NT,Admission.Inner.config,Nat.mul_assoc] using
    outputs.final enabled _ _ _ hb.1 hb.2.1 hb.2.2

def prefixLayout (r : Admission.Prefix) (BS : Nat) := HostShape.cumulative r.T r.S r.BT BS
def prefixInput (r : Admission.Prefix) (before : Memory) (scratch : BlockState) : BlockState :=
  {scratch with mem:=fun region i=>if region="s" then
    MemCell.real (CumulativeHost.materialized (prefixLayout r 16) r.T r.S
      (logical before r.input r.T r.S) i) else scratch.mem region i}
def Prefix (r : Admission.Prefix) (before : Memory) (i : Nat) : ℝ :=
  Host.chunkPrefix r.BT (logical before r.input r.T r.S (i/(r.T*r.S))) ((i/r.S)%r.T) (i%r.S)

theorem prefix_exists (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : Memory) (scratch : BlockState) :
    ∃ state,DirectCumulativeSequence.Runs (prefixLayout r BS) (r.B*r.H) "s" "o" Bool.false .fp32 .fp32
      (DirectCumulativeWhole.plans (prefixLayout r BS) (r.B*r.H))
      {prefixInput r before scratch with undef:=fun _ _=>0} state ∧
      (∀ h t f,h<r.B*r.H → t<r.T → f<r.S →
        state.readMem "o" (h*(HostFP.padded r.T r.BT*HostFP.padded r.S 64)+t*HostFP.padded r.S 64+f)=
          Host.chunkPrefix r.BT (logical before r.input r.T r.S h) t f) ∧
      BlockState.WriteWithin (DirectCumulativeSequence.allWrites (prefixLayout r BS) (r.B*r.H)
        (DirectCumulativeWhole.plans (prefixLayout r BS) (r.B*r.H)) "o" (prefixInput r before scratch))
        (prefixInput r before scratch) state := by
  have bt : 0<r.BT:=by have:=((r.check_iff).mp checked).1.1;omega
  have result:=DirectCumulativeWhole.executes (prefixLayout r BS) (r.B*r.H) r.T r.S "s" "o"
    Bool.false .fp32 .fp32 (logical before r.input r.T r.S)
    (HostShape.cumulative_canonical _ _ _ _ bt selected) selected
    (HostShape.le_padded _ _ bt) (HostShape.le_padded _ _ (by decide)) (by decide)
    (prefixInput r before scratch) (by
      intro off
      simp [prefixInput,prefixLayout,CumulativeHost.materialized,HostShape.cumulative,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
  simpa [prefixLayout,HostShape.cumulative,Nat.mul_assoc] using result

def prefixResult (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : Memory) (scratch : BlockState) : BlockState :=
  (prefix_exists r checked BS selected before scratch).choose

theorem prefix_value (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : Memory) (scratch : BlockState)
    (i : Nat) (hi : i<prefixCount r) :
    (prefixResult r checked BS selected before scratch).readMem "o" (HostData.prefixOffset r i)=Prefix r before i := by
  have hb:=HostCoordinates.decode3 hi
  have outputs:=(prefix_exists r checked BS selected before scratch).choose_spec.2.1
  simpa [prefixResult,HostData.prefixOffset,Prefix,Nat.mul_assoc] using outputs _ _ _ hb.1 hb.2.1 hb.2.2

#specNonCircular History avoiding [innerResult,exec]
#specNonCircular Final avoiding [innerResult,exec]
#specNonCircular Prefix avoiding [prefixResult,exec]
#axiomsClean inner_exists
#axiomsClean history
#axiomsClean final
#axiomsClean prefix_exists
#axiomsClean prefix_value
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RealSource
