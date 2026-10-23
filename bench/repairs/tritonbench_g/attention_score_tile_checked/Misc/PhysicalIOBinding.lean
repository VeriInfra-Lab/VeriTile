import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.RealIOProof
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIO
import VeriTile.Spec

/-! Optional binding of the original finite word-indexed physical reader to
the public source IO. Read-only aliases use the same reader and ABI, without
adding separation or copying. The old zero-output initialization is explicit. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.PhysicalIOBinding
open VeriTile Triton FP.Structural Reused Geometry Physical NativeInstructions NativeStaging
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def realInputs (c : Shape) (abi : ABI c) (physical : Nat → ℝ) : RealInterface.Inputs ℝ c :=
  fun i j => if i.val=3 then 0 else physical ((abi.alloc (RealInterface.region i)).base+j.val)

theorem real_binding (c : Shape) (abi : ABI c) (physical : Nat → ℝ)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    (RealInterface.programIO c offset size scale window complement).InputBinding
      (RealReturned.zeroOutput c (RealPhysical.initial abi physical)) (realInputs c abi physical) := by
  intro i j active
  change Fin 4 at i
  fin_cases i
  all_goals
    simp [RealInterface.programIO,RealInterface.arity,RealInterface.region,
      ChanTy.read,BlockState.readMem,RealReturned.zeroOutput,RealPhysical.initial,realInputs,j.isLt]

theorem real_empty_input (c : Shape) (abi : ABI c) (physical : Nat → ℝ) :
    RealInterface.initialOutput c (realInputs c abi physical) := by
  intro _ j
  rfl

theorem real_normalized (c : Shape) (abi : ABI c) (physical : Nat → ℝ) :
    Normalized c abi physical (RealInterface.logicalInputs c (realInputs c abi physical)) := by
  constructor
  · intro z h i d
    simp [RealInterface.logicalInputs,realInputs,RealInterface.region,Nat.add_assoc]
  · intro z h j d
    simp [RealInterface.logicalInputs,realInputs,RealInterface.region,Nat.add_assoc]
  · intro z h i
    simp [RealInterface.logicalInputs,realInputs,RealInterface.region,Shape.normalizerBase,Nat.add_assoc]

def nativeInputs (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32) : NativeInterface.Inputs c := by
  intro i
  change Fin 4 at i
  exact match i with
    | ⟨0,_⟩ => fun j => .bits (physical ((abi.alloc "q").base+j.1.val))
    | ⟨1,_⟩ => fun j => .bits (physical ((abi.alloc "k").base+j.1.val))
    | ⟨2,_⟩ => fun j => .bits (physical ((abi.alloc "m").base+j.1.val))
    | ⟨3,_⟩ => fun _ => NativeFormula.zero32

theorem native_binding (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    (NativeInterface.programIO c offset size scale window complement).InputCells
      (nativeInputs c abi physical)
      (FPReturned.zeroOutput algebra c (Physical.initial abi (NativePhysical.input physical))) := by
  intro i j active
  change Fin 4 at i
  fin_cases i
  · change (FPReturned.zeroOutput algebra c (Physical.initial abi (NativePhysical.input physical))).mem "q" j.1.val = Cell.mk .real (NativeInterface.encode c (.bits (physical ((abi.alloc "q").base+j.1.val))))
    simp [FPReturned.zeroOutput,Physical.initial,FPStorage.state,FPStorage.importMemory,FPStorage.cell,FPStorage.project,NativePhysical.input,NativeInterface.encode]
  · change (FPReturned.zeroOutput algebra c (Physical.initial abi (NativePhysical.input physical))).mem "k" j.1.val = Cell.mk .real (NativeInterface.encode c (.bits (physical ((abi.alloc "k").base+j.1.val))))
    simp [FPReturned.zeroOutput,Physical.initial,FPStorage.state,FPStorage.importMemory,FPStorage.cell,FPStorage.project,NativePhysical.input,NativeInterface.encode]
  · change (FPReturned.zeroOutput algebra c (Physical.initial abi (NativePhysical.input physical))).mem "m" j.1.val = Cell.mk .real (NativeInterface.encode c (.bits (physical ((abi.alloc "m").base+j.1.val))))
    simp [FPReturned.zeroOutput,Physical.initial,FPStorage.state,FPStorage.importMemory,FPStorage.cell,FPStorage.project,NativePhysical.input,NativeInterface.encode]
  · change (FPReturned.zeroOutput algebra c (Physical.initial abi (NativePhysical.input physical))).mem "out" j.1.val = Cell.mk .real (NativeInterface.encode c NativeFormula.zero32)
    have empty : RealInterface.empty c := active.resolve_left (by simp)
    simp [FPReturned.zeroOutput,NativeInterface.encode,empty,j.1.isLt,
      NativeFormula.zero32,FPContext.zero,algebra]

theorem native_empty_input (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32) :
    NativeInterface.initialOutput c (nativeInputs c abi physical) := by
  intro _ j
  rfl

/-- Actual flat source allocations supply bounds for every admitted IO value. -/
theorem real_bounds (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (xs : RealInterface.Inputs ℝ c) (s : BlockState) :
    (RealInterface.programIO c offset size scale window complement).InBounds
      (DenseSafety.bounds c) s xs := by
  constructor
  · intro i j _
    fin_cases i <;> exact j.isLt
  constructor
  · intro o j _
    simpa [RealInterface.programIO,DenseSafety.bounds] using j.isLt
  · intro i
    exact Fin.elim0 i

theorem native_bounds (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (xs : NativeInterface.Inputs c) :
    (NativeInterface.programIO c offset size scale window complement).InBounds
      (DenseSafety.bounds c) xs := by
  constructor
  · intro r outside
    simp only [NativeInterface.programIO,List.mem_cons,List.not_mem_nil,or_false,not_or] at outside
    simp [DenseSafety.bounds,outside.1,outside.2.1,outside.2.2.1,outside.2.2.2]
  constructor
  · intro i j _
    change Fin 4 at i
    fin_cases i <;> simpa [NativeInterface.programIO,NativeInterface.inputPorts,DenseSafety.bounds] using j.1.isLt
  constructor
  · intro o j _
    change Fin 1 at o
    fin_cases o
    simpa [NativeInterface.programIO,NativeInterface.outputPorts,DenseSafety.bounds] using j.1.isLt
  · intro i
    exact Fin.elim0 i

#axiomsClean real_binding
#axiomsClean native_binding
#axiomsClean real_bounds
#axiomsClean native_bounds
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.PhysicalIOBinding
