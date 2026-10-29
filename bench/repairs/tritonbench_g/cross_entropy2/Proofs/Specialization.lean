import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledReturns

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
open Source Reduction
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
noncomputable section

/-- Normalized compiler attributes for the pinned annotated forward kernel.
Pointers are byte addresses including storage offsets. Scalar floating values
are runtime FP32 arguments and are not specialized. Empty attribute entries
and absent entries are equivalent; the runtime cache key may still differ. -/
structure JITKey where
  pointerDiv16 : List Bool
  ignoredDiv16 : Bool
  totalDiv16 : Bool
  startDiv16 : Bool
  colsDiv16 : Bool
  rowsDiv16 : Bool
  strideDiv16 : Bool
  width : Nat
  smoothing : Bool
  split : Bool
  deriving DecidableEq

def alignedKey : JITKey :=
  ⟨[Bool.true,Bool.true,Bool.true,Bool.true,Bool.true],Bool.false,
    Bool.true,Bool.true,Bool.true,Bool.true,Bool.true,65536,Bool.true,Bool.true⟩
def offset4Key : JITKey :=
  {alignedKey with pointerDiv16 := [Bool.true,Bool.true,Bool.true,Bool.false,Bool.false]}

def bytePointer {σ : Type} (layout : Layout σ) (state : World Compiled σ)
    (region : VeriTile.Triton.RegionName) (offset : Nat) : Nat :=
  (layout.buffer state.native region).base+4*offset

/-- The metadata key uses the actual prepared allocations and views. It is
not inferred from logical contiguity alone: an offset4 contiguous view is a
different key. This is the ordinary JIT divisibility contract checked by the
offline runtime binder probe, not a compiler implementation proof. -/
def keyFor {σ : Type} (layout : Layout σ) (state : World Compiled σ) (p : Parameters) : JITKey where
  pointerDiv16 :=
    [(decide (bytePointer layout state p.loss p.lossBase%16=0)),
     (decide (bytePointer layout state p.lse p.lseBase%16=0)),
     (decide (bytePointer layout state p.zLoss p.zLossBase%16=0)),
     (decide (bytePointer layout state p.logits p.logitsBase%16=0)),
     (decide (bytePointer layout state (VeriTile.Triton.Region.cast p.labels) p.labelsBase%16=0))]
  ignoredDiv16 := decide (p.ignored%16=0)
  totalDiv16 := decide (p.totalClasses%16=0)
  startDiv16 := decide (p.classStart%16=0)
  colsDiv16 := decide (p.cols%16=0)
  rowsDiv16 := decide (p.rows%16=0)
  strideDiv16 := decide (p.logitsStride%16=0)
  width := p.block
  smoothing := p.hasSmoothing
  split := p.split

/-- Only the two actually traced JIT keys are registered here. Other admitted
metadata are not narrowed out of the API and require another profile binding. -/
def selectProfile (key : JITKey) : Option DeviceSchedule :=
  if key=alignedKey then some .aligned65536
  else if key=offset4Key then some .pureTrees else none

def deviceTrees (schedule : DeviceSchedule) (host : Profile) : Profile :=
  {host with
    sumTree := fun width positive => if equal : width=65536 then by
      subst width
      exact if schedule=.aligned65536 then sm90AlignedLarge else sm90Large
      else host.sumTree width positive
    maxTree := fun width positive => if equal : width=65536 then by
      subst width
      exact if schedule=.aligned65536 then sm90AlignedLarge else sm90Large
      else host.maxTree width positive}

/-- Explicit ordinary compiler metadata contract. It states selection of
instruction syntax and pure reduction plans, never equality of numerical
reductions, whole executions or outputs. The two clauses are backed by the
pinned SM90/Triton 3.7.1, four-warp, default-fusion traces. General keys remain
outside this finite binding; host trees are separately supplied profile data. -/
structure CompilerSelection (actual : JITKey → DeviceSchedule) : Prop where
  aligned : actual alignedKey=.aligned65536
  offset4 : actual offset4Key=.pureTrees

theorem selected_aligned : selectProfile alignedKey=some .aligned65536 := by
  simp [selectProfile]
theorem selected_offset4 : selectProfile offset4Key=some .pureTrees := by
  simp [selectProfile,offset4Key,alignedKey]

theorem aligned_key_fields {σ : Type} (layout : Layout σ) (state : World Compiled σ)
    (p : Parameters) (key : keyFor layout state p=alignedKey) :
    p.block=65536 ∧ p.hasSmoothing=Bool.true ∧ p.split=Bool.true ∧ p.cols%4=0 := by
  have width := congrArg JITKey.width key
  have smoothing := congrArg JITKey.smoothing key
  have split := congrArg JITKey.split key
  have cols : decide (p.cols%16=0)=Bool.true := congrArg JITKey.colsDiv16 key
  have aligned : p.cols%16=0 := of_decide_eq_true cols
  exact ⟨width,smoothing,split,by omega⟩

theorem aligned_metadata_mask {σ : Type} (layout : Layout σ) (state : World Compiled σ)
    (p : Parameters) (key : keyFor layout state p=alignedKey) (block lane slot : Nat) :
    (block*p.block+alignedCoordinate lane slot<p.cols) ↔
      (block*p.block+4*lane+512*(slot/4)<p.cols) := by
  have fields := aligned_key_fields layout state p key
  apply aligned_shared_mask _ _ _ _ fields.2.2.2
  simp [fields.1,Nat.mul_mod]

/-- Concrete key binding composes with source-address lowering and scalar
interpretation. No numerical reduction result is a premise. -/
theorem aligned_source_scalar {σ : Type} (layout : Layout σ) (state : World Compiled σ)
    (n : Ports) (profile : Profile) (p : Parameters) (row block : Nat)
    (key : keyFor layout state p=alignedKey) :
    let active := fun lane : Fin p.block =>
      decide (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols)
    let input := fun lane : Fin p.block =>
      ((state.memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real).value
    decode n profile ((values (algebra .aligned65536 p.logitScale) p state.memory row block).sumLogits.getD
      ((algebra .aligned65536 p.logitScale).literal 0))=
      alignedSmoothing.run n (n.literal p.logitScale)
        (fun i => runInstruction n profile (boundedInput active input i)) (boundedActive active) := by
  have fields := aligned_key_fields layout state p key
  exact source_smoothing_scalar n profile p state.memory row block fields.1 fields.2.1

#axiomsClean selected_aligned
#axiomsClean selected_offset4
#axiomsClean aligned_metadata_mask
#axiomsClean aligned_source_scalar
#specNonCircular CompilerSelection avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Scheduled
