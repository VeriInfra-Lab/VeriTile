import bench.tritonbench_g.int8_dequant_matmul.Proofs.DirectWords
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Ingress
import bench.tritonbench_g.int8_dequant_matmul.Kernels

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
open Direct
noncomputable section

/-- The original checked input domain, unchanged. Unit dimensions, arbitrary
INT32 overflow, nonfinite scales, offsets and read-only same-dtype aliases remain. -/
abbrev InputContract := Direct.InputContract

/-- One prepared kernel call; allocation and autotuning are outside this task.
Every original configuration remains admitted. Freshness is input/output
storage separation, not a postcondition about the answer. -/
structure Prepared (r : Request) (out : Nat) (c : Config) : Prop where
  admitted : Source.Admitted c
  fresh : (requestBuffers r out).Fresh

/-- The source uses atomic FP16 addition for split K. The existing zero hook
establishes this input state before a kernel call; this task does not rerun it. -/
def Zeroed (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (initial : Memory F H) : Prop :=
  c.S ≠ 1 → ∀ cell : MatrixCell s,
    initial (outputAddress s b cell)=.inr (.inr a.hzero)

theorem zeroed_read (zero : Zeroed a s c b initial) (split : c.S≠1) (cell : MatrixCell s) :
    read16 a initial (outputAddress s b cell)=a.hzero := by
  simp [read16,zero split cell]

def typedJIT (r : Request) (c : Config) : VeriTile.Triton.ComputeKernel :=
  TypedSource.kernel "A" "B" "C" "bias" "state_x" "state_w"
    (requestShape r).M (requestShape r).N (requestShape r).K
    (requestShape r).K 1 (requestShape r).N 1 (requestShape r).N 1
    c.BM c.BN c.BK 8 c.S (steps (requestShape r) c)
    ((requestShape r).K % (c.BK*c.S)==0) (hasBias r)

theorem prepared_bounds (r : Request) (out : Nat) (c : Config) (p : Prepared r out c)
    (hm : 0<(requestShape r).M) (hn : 0<(requestShape r).N) :
    IndexABI (requestShape r) c (requestBuffers r out)
      (canonical (requestShape r).M (requestShape r).K)
      (canonical (requestShape r).K (requestShape r).N)
      (canonical (requestShape r).M (requestShape r).N) ∧ GridABI (requestShape r) c :=
  native_admitted r.a.info r.b.info r.x.info r.w.info (r.bias.map TensorView.info)
    out r.contract hm hn c p.admitted

/-- Both dynamic integer widths retain exact signed value through the cast. -/
theorem runtime_cast_exact (x : RuntimeIndex) : x.widen.toInt=x.decode :=
  RuntimeIndex.exact x

theorem constexpr_one_cast : (BitVec.ofNat 64 1).toInt=1 := by decide

#axiomsClean prepared_bounds
#axiomsClean runtime_cast_exact
#axiomsClean constexpr_one_cast
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
