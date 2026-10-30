import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Prefix
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Arithmetic
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Lowering

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
open scoped Classical
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def activeK (p : Parameters) (step lane : Nat) := step*64+lane<p.inner
def halfZero {α : Type} (a : Arithmetic α) : α :=
  a.cast none .real .fp16 (a.literal none .real 0)
def inputValue {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [16,64]) : α :=
  if activeK p step i.2.1.val then (s.mem "A" (aOffset p s step i)).read .fp16 else halfZero a
def rmsByte {α : Type} [Inhabited α] (p : Parameters) (s : State α) (step lane : Nat) : Int :=
  if activeK p step lane then (s.mem "RMS" (step*64+lane)).read .int else 0
def weightByte {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (weight : RegionName) (step : Nat) (i : TileIndex [64,16]) : Int :=
  if activeK p step i.1.val then (s.mem weight (wOffset p s step i)).read .int else 0
def rmsValue {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (step lane : Nat) : α :=
  if p.byte then a.cast none .f8e5 .fp16 (a.decode .f8e5 (Float8E5Bits.ofInt (rmsByte p s step lane)))
  else if activeK p step lane then (s.mem "RMS" (step*64+lane)).read .fp16 else halfZero a
def weightValue {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (weight : RegionName) (step : Nat) (i : TileIndex [64,16]) : α :=
  if p.byte then a.cast none .real .fp16
      (a.cast (some .fp32) .f8e5 .real (a.decode .f8e5 (Float8E5Bits.ofInt (weightByte p s weight step i))))
  else if activeK p step i.1.val then (s.mem weight (wOffset p s step i)).read .fp16 else halfZero a
def nativeWeightValue {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (weight : RegionName) (step : Nat) (i : TileIndex [64,16]) : α :=
  if p.byte then a.cast none .f8e5 .fp16
    (a.decode .f8e5 (Float8E5Bits.ofInt (weightByte p s weight step i)))
  else if activeK p step i.1.val then (s.mem weight (wOffset p s step i)).read .fp16 else halfZero a
def scaledValue {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [16,64]) : α :=
  a.binary none .fp16 .mul (inputValue a p s step i) (rmsValue a p s step i.2.1.val)

/-- Independent tile recurrence from initial input cells. `native` selects
the explicitly displayed live-carry K16 instruction chain. -/
def accumulation {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State α) (weight : RegionName) : Nat → TileIndex [16,16] → α
  | 0 => fun _ => a.literal (some .fp32) .real 0
  | step+1 => fun i =>
    if native then tensor a (batch := []) (scaledValue a p s step) (nativeWeightValue a p s weight step)
      (accumulation a native p s weight step) i
    else a.binary (some .fp32) .real .add (accumulation a native p s weight step i)
      (a.cast (some .fp32) .fp32 .real
        (tensor a (batch := []) (scaledValue a p s step) (weightValue a p s weight step)
          (fun _ => a.literal (some .fp32) .real 0) i))
def moment {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) : Nat → TileIndex [16,64] → α
  | 0 => fun _ => a.literal (some .fp32) .real 0
  | step+1 => fun i => a.binary (some .fp32) .real .add (moment a p s step i)
      (a.powi (some .fp32) (a.cast (some .fp32) .fp16 .real (inputValue a p s step i)) 2)

structure Invariant {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial : State α) (step : Nat) (current : State α) : Prop extends
    Indices p initial step current where
  acc1 : current.regs .real [16,16] "acc1" = some (accumulation a native p initial "W1" step)
  acc2 : current.regs .real [16,16] "acc2" = some (accumulation a native p initial "W3" step)
  sum : current.regs .real [16,64] "a_sum" = some (moment a p initial step)

def ByteValues {α : Type} [Inhabited α] (s : State α) : Prop :=
  ∀ weight ∈ (["RMS","W1","W3"] : List RegionName), ∀ offset,
    Float8E5Bits.isInt8 ((s.mem weight offset).read .int) = Bool.true

theorem byte_rms {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (ok : ByteValues s) (step lane : Nat) :
    Float8E5Bits.isInt8 (rmsByte p s step lane) = Bool.true := by
  by_cases h : activeK p step lane
  · simp only [rmsByte,if_pos h]; exact ok "RMS" (by simp) _
  · simp [rmsByte,h,Float8E5Bits.isInt8]
theorem byte_weight {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (ok : ByteValues s) (step : Nat) (weight : RegionName)
    (hw : weight="W1" ∨ weight="W3") (i : TileIndex [64,16]) :
    Float8E5Bits.isInt8 (weightByte p s weight step i) = Bool.true := by
  by_cases h : activeK p step i.1.val
  · simp only [weightByte,if_pos h]
    apply ok weight
    rcases hw with rfl|rfl <;> simp
  · simp [weightByte,h,Float8E5Bits.isInt8]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
