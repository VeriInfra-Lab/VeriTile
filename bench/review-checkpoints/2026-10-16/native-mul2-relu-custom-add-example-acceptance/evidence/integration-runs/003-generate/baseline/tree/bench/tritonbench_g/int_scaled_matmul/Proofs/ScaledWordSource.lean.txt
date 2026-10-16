import bench.tritonbench_g.int_scaled_matmul.IntScaledMatmul
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.IntegerConversionTotal
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedVariables false
def typedScaled
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool)
    (numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.region s1_ptr = tl.float32, c_ptr = tl.int32
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = tl.arange(0, $(BLOCK_K))
  A = $((a_ptr : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((b_ptr : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for j in range(0, $(numKBlocks), $(1)) {
    k = $(K) - j * $(BLOCK_K)
    if EVEN_K {
      a = tl.load(A)
      b = tl.load(B)
    } else {
      a = tl.load(A, mask=rk[None, :] < k, other=0.0)
      b = tl.load(B, mask=rk[:, None] < k, other=0.0)
    }
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  idx_m = rm[:, None]
  idx_n = rn[None, :]
  mask = (idx_m < $(M)) & (idx_n < $(N))
  xindex = idx_n + ($(N) * idx_m)
  tmp0 = tl.load(s1_ptr + (tl.broadcast_to(idx_m, [$(BLOCK_M), $(BLOCK_N)])), mask, eviction_policy="evict_last").to(tl.float32)
  tl.store(c_ptr + (tl.broadcast_to(xindex, [$(BLOCK_M), $(BLOCK_N)])), (acc * tmp0).to(tl.int32), mask)
}

structure Parameters where
  (input weight output scale : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride scaleRowStride scaleColumnStride
    blockRows blockColumns blockInner groupRows steps : Nat)
  (even : Bool)
def original (p : Parameters) : ComputeKernel := typedScaled
  (Region.cast p.input) (Region.cast p.weight) p.output p.scale p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleRowStride p.scaleColumnStride
  p.blockRows p.blockColumns p.blockInner p.groupRows p.even p.steps

def widthAssignment : ComputeStmt → ComputeStmt
  | .assign .int sh name (.alg e) =>
    if name = "a" ∨ name = "b" then .assign .int sh name (.alg (.intToSigned 8 e))
    else if name = "acc" then .assign .int sh name (.alg (.intToSigned 32 e))
    else .assign .int sh name (.alg e)
  | stmt => stmt
/-- Both EVEN_K branches retain byte storage semantics; every int32
accumulator update wraps, without a no-overflow premise. -/
def widths : ComputeStmt → ComputeStmt
  | .ifThenElse cond yes no => .ifThenElse cond (yes.map widthAssignment) (no.map widthAssignment)
  | stmt => widthAssignment stmt

def prelude (p : Parameters) := (original p).surfaceBody.take 16
def originalBody (p : Parameters) : List ComputeStmt := match (original p).surfaceBody[16]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def body (p : Parameters) := (originalBody p).map widths
def epilogue (p : Parameters) := (original p).surfaceBody.drop 17

def source (p : Parameters) : ComputeKernel := .mk [p.input,p.weight,p.scale] [p.output]
  (prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p)
def wrap (w : Nat) (x : Int) : Int := (BitVec.ofInt w x).toInt

abbrev Word := FP.Value .fp32
local instance scaledSourceDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- fp32 scalar operations and width-checked signed conversion, with an
arbitrary fallback only where the source integer conversion is undefined. -/
def backend (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) : Algebra Word :=
  let scalar := FP.Software.scalarFP32 rest cfg
  let converted := FP.IntegerConversion.fp32TotalBackend scalar u (fun _ _ => 0)
  { converted with
    fromInt := fun p n => if p=some .fp32 then FP.round .fp32 cfg (n : Rat) else converted.fromInt p n
    integerDot := fun _ {batch} {_} {_} {_} a b =>
      some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt) }

theorem original_decomposition (p : Parameters) : (original p).surfaceBody =
    prelude p ++ [.forRange "j" 0 p.steps 1 (originalBody p)] ++ epilogue p := by rfl
#axiomsClean original_decomposition
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
