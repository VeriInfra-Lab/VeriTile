import bench.repairs.tritonbench_g.rotary_transform.Proofs.Views
import bench.repairs.tritonbench_g.rotary_transform.Proofs.Metadata
namespace VeriTile.Bench.Repairs.RotaryTransform.Physical
open Native

def callerConfiguration (s : Shape) (a : MetadataABI) (v : InputViews) (m : Bytes) : Config :=
  ⟨s,canonicalMetadata s
    ⟨fun i => integer m (.other 0) a.cumulative64 (i*a.cumulativeStride),
     fun i => integer m (.other 1) a.offsets64 (i*v.offsetsStride)⟩⟩

theorem staged_offset_integer (s : Shape) (a : MetadataABI) (v : InputViews) (m : Bytes) (i : Nat) :
    integer (stageInputs s v a.offsets64 m) .offsets a.offsets64 i=
      integer m (.other 1) a.offsets64 (i*v.offsetsStride) := by
  unfold integer
  cases h : a.offsets64
  · simp only [Bool.false_eq_true,↓reduceIte]
    rw [stage_offset32]
  · simp only [↓reduceIte]
    rw [show 2*i+1=2*i+(if true then 1 else 0) from rfl,
      stage_offset64 s v m i true,
      show 2*i=2*i+(if false then 1 else 0) from (Nat.add_zero _).symm,
      stage_offset64 s v m i false]
    rfl

theorem staged_configuration (s : Shape) (a : MetadataABI) (v : InputViews) (m : Bytes) :
    originalConfiguration s a (stageInputs s v a.offsets64 m)=callerConfiguration s a v m := by
  unfold originalConfiguration callerConfiguration
  apply congrArg (fun md => Config.mk s (canonicalMetadata s md))
  unfold originalMetadata
  apply congrArg₂ Metadata.mk
  · funext i
    simp [integer,load,stageInputs]
  · funext i
    exact staged_offset_integer s a v m i

/-- Logical values read directly from caller allocations. Padding bytes and
strides remain in the physical input; only coefficient addressing changes. -/
def callerWords (s : Shape) (v : InputViews) (m : Bytes) : Memory Word := fun r a =>
  if r=.cosine then load m (.other 2) (v.cosineAt s a)
  else if r=.sine then load m (.other 3) (v.sineAt s a)
  else load m r a

theorem staged_inputs (s : Shape) (v : InputViews) (wide : Bool) (m : Bytes) :
    (∀ a,load (stageInputs s v wide m) .input a=callerWords s v m .input a) ∧
    (∀ a,load (stageInputs s v wide m) .cosine a=callerWords s v m .cosine a) ∧
    (∀ a,load (stageInputs s v wide m) .sine a=callerWords s v m .sine a) := by
  constructor
  · intro a; simp [stage_input,callerWords]
  constructor
  · intro a; simp [stage_cosine,callerWords]
  · intro a; simp [stage_sine,callerWords]

#specNonCircular callerConfiguration avoiding [source,pairValues,dynamic,native]
#specNonCircular callerWords avoiding [source,pairValues,dynamic,native]
#axiomsClean staged_configuration
#axiomsClean staged_inputs
end VeriTile.Bench.Repairs.RotaryTransform.Physical
