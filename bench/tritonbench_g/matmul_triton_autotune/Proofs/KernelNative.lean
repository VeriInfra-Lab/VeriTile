import bench.tritonbench_g.matmul_triton_autotune.Kernels
import VeriTile.Triton.Float.ScalarBackend
import bench.tritonbench_g.quantize_kv_transform.Sources.Mathematical_MixedBits

/-! Concrete scalar FP32 and FP16 semantics, plus fixed instruction atoms for
the pinned SM80 TF32 MMA and gfx942 XF32 MFMA profiles. These atoms expose the
documented implementation-dependent dot rounding; they are not a variable-K
dot oracle or an assumed whole-result equality. Compiler/hardware conformance
to this named interpretation is a separate model boundary. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Floating
open VeriTile.Triton FP.Structural Whole
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 3000000
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Per-output-lane instruction arithmetic, including special values and
operand interpretation of the original FP32 words. Lane/register mapping and
selection of these opcodes are part of the explicit lowering boundary. -/
structure Instructions where
  cudaMma8 : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
  hipMfma4 : (Fin 4 → Word) → (Fin 4 → Word) → Word → Word

def zero : Word := FP.Value.zero .fp32
def slope : Word := FP.Value.ofBits .fp32 0x3c23d70a
def narrow : Word → Word := QuantizeKvTransform.MixedBits.narrow .ieeeValue
def nonnegative (x : Word) : Bool := FP.lt .ieeeValue zero x || FP.eq .ieeeValue zero x

/-- Unreachable scalar cases are fixed, not supplied by the caller. All
floating operations actually used by the quotation are overridden below. -/
def unused : Algebra Word where
  literal := fun _ _ _ => zero
  negInf := FP.Value.infinity .fp32 Bool.true
  binary := fun _ _ _ _ _ => zero
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => zero
  fromInt := fun _ _ => zero
  fp32Bits := fun x => ⟨x.bits⟩
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => zero
  reduceSum := fun _ _ _ _ _ _ => zero

/-- Source literals 0 and 0.01, ordered FP32 multiplication and exact binary16
rounding/packing. All 2^32 FP32 words remain admitted. -/
def scalar : Algebra Word :=
  { FP.Software.scalarFP32 unused .ieeeValue with
    literal := fun _ _ x => if x = 0.01 then slope else zero
    cast := fun _ src dst x => match src, dst with
      | .real, .fp16 => narrow x
      | _, _ => x }

def width : Config → Nat | .cuda128 | .cuda64 => 8 | .hip128 => 4

def block (instructions : Instructions) (config : Config) (k : Nat)
    (a b : Fin k → Word) (carry : Word) : Word :=
  match config with
  | .cuda128 | .cuda64 =>
    (List.range (k/8)).foldl (fun acc group => instructions.cudaMma8
      (fun lane => if h : group*8+lane.val < k then a ⟨group*8+lane.val,h⟩ else zero)
      (fun lane => if h : group*8+lane.val < k then b ⟨group*8+lane.val,h⟩ else zero) acc) carry
  | .hip128 =>
    (List.range (k/4)).foldl (fun acc group => instructions.hipMfma4
      (fun lane => if h : group*4+lane.val < k then a ⟨group*4+lane.val,h⟩ else zero)
      (fun lane => if h : group*4+lane.val < k then b ⟨group*4+lane.val,h⟩ else zero) acc) carry

theorem block_exact (config : Config) : config.bk / width config * width config = config.bk := by
  cases config <;> decide

theorem instruction_lane_bound (config : Config) (group : Nat)
    (hg : group < config.bk / width config) (lane : Fin (width config)) :
    group * width config + lane.val < config.bk := by
  have hl := lane.isLt
  have he := block_exact config
  nlinarith

def dotValue (instructions : Instructions) (config : Config) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word)
    (acc : TileIndex (batch++[m,n]) → Word) : TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => block instructions config k
      (fun e => a (i.1,e,PUnit.unit)) (fun e => b (e,i.2.1,PUnit.unit)) (acc i)
  | _::tail => fun i => dotValue instructions config (batch:=tail)
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) (fun j => acc (i.1,j)) i.2
termination_by batch.length

def dot (instructions : Instructions) (config : Config) : FPValue.DotAcc Word where
  apply := fun _ _ _ _ _ {batch} {m} {k} {n} a b acc => dotValue instructions config a b acc
  le := fun _ _ x y => FP.lt .ieeeValue x y || FP.eq .ieeeValue x y

def numerics (instructions : Instructions) (config : Config) : Numerics Word where
  zero := zero
  dotAcc := block instructions config
  nonnegative := nonnegative
  slopeMul := fun x => FP.mul .ieeeValue slope x
  castHalf := narrow

/-- The former six primitive obligations are discharged by the selected
definitions. No primitive correspondence, finite operand condition or
numerical experiment premise is required by the public theorem. -/
theorem primitives (instructions : Instructions) (config : Config) :
    PrimitiveLaws scalar (dot instructions config) (numerics instructions config) := by
  constructor
  · norm_num [scalar, numerics, zero]
  · norm_num [scalar, numerics, zero]
  · intros; simp only [dot, dotValue, numerics]
  · intro x; norm_num [dot, scalar, numerics, nonnegative, zero]
  · intro x
    simp [scalar, numerics, FP.Software.scalarFP32, FP.Software.fp32Context]
  · intro x; rfl

#axiomsClean block_exact
#axiomsClean instruction_lane_bound
#axiomsClean primitives
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Floating
