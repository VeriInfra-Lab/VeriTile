import bench.tritonbench_g.parallel_attention.Proofs.FPForwardWhole
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialSum sequentialFP32)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

/-- Explicit software dot: ascending K, positive-zero accumulator, rounded
fp32 multiply followed by rounded fp32 addition at each element. It does not
claim to be the GPU's reduction or FMA profile. Batch dimensions are preserved. -/
def sequentialDot (cfg : FP.Config) : (batch : TileShape) → {m k n : Nat} →
    Values (FP.Value .fp32) .real (batch++[m,k]) → Values (FP.Value .fp32) .real (batch++[k,n]) →
    Values (FP.Value .fp32) .real (batch++[m,n])
  | [], _, k, _, a, b => fun i => (List.finRange k).foldl
      (fun total j => FP.add cfg total (FP.mul cfg (a ⟨i.1,j,()⟩) (b ⟨j,i.2.1,()⟩))) (FP.Value.zero .fp32)
  | _::rest, _, _, _, a, b => fun i => sequentialDot cfg rest (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def softwareDot (cfg : FP.Config) : DotPrimitive (FP.Value .fp32) where
  apply := fun _ _ {batch} {_} {_} {_} a b => sequentialDot cfg batch a b

/-- Concrete fp32 payloads for both active and unspecified source lanes. -/
def wordMatrix [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32)) (R : RegionName)
    (base rows cols rowStride colStride rowOffset colOffset m n : Nat) : TileIndex [m,n] → FP.Value .fp32 := fun i =>
  let row := rowOffset+i.1.val
  let col := colOffset+i.2.1.val
  let address := base+row*rowStride+col*colStride
  if row<rows ∧ col<cols then (s.mem R address).read .real else s.undef .real R address

def one32 : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000

def roundedQuery [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) : TileIndex [c.BTL,c.BK] → FP.Value .fp32 := fun i =>
  FP.mul cfg (wordMatrix s c.q (s.pids 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    (queryChunk c s) (keyChunk c s) c.BTL c.BK i) one32

/-- An independent bit-value recurrence for scale=1. Every arithmetic
operation below is FP.add/FP.mul or an explicitly ordered fp32 reduction. -/
def roundedForwardUpdate [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (c : ForwardConfig) (s : State (FP.Value .fp32)) (diagonal : Bool) (start : Nat)
    (previous : ForwardAccumulator (FP.Value .fp32) c.BTL c.BV) : ForwardAccumulator (FP.Value .fp32) c.BTL c.BV :=
  let key := wordMatrix s c.k (s.pids 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    (keyChunk c s) start c.BK c.BTS
  let value := wordMatrix s c.v (s.pids 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    start (valueChunk c s) c.BTS c.BV
  let raw := sequentialDot cfg [] (roundedQuery cfg c s) key
  let score := fun (i : TileIndex [c.BTL,c.BTS]) =>
    let squared := FP.mul cfg (raw i) (raw i)
    if diagonal then if start-queryChunk c s+i.2.1.val ≤ i.1.val then squared else FP.Value.zero .fp32
    else squared
  let total := sequentialSum cfg (shape := [c.BTL,c.BTS]) ⟨1,by simp⟩ Bool.false score
  let contribution := sequentialDot cfg [] score value
  ⟨fun i => FP.add cfg (previous.output i) (contribution i),
   fun i => FP.add cfg (previous.normalizer i) (total i)⟩

def roundedForwardValues [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) : ForwardAccumulator (FP.Value .fp32) c.BTL c.BV :=
  let initial : ForwardAccumulator (FP.Value .fp32) c.BTL c.BV :=
    ⟨fun _ => FP.Value.zero .fp32,fun _ => FP.Value.zero .fp32⟩
  let first := foldRange (roundedForwardUpdate cfg c s Bool.false) 0 (queryChunk c s) c.BTS initial
  foldRange (roundedForwardUpdate cfg c s Bool.true) (queryChunk c s) ((s.pids 1+1)*c.BTL) c.BTS first

def roundedOutputMemory [Inhabited (FP.Value .fp32)] (c : ForwardConfig)
    (s : State (FP.Value .fp32)) (values : ForwardAccumulator (FP.Value .fp32) c.BTL c.BV) : State (FP.Value .fp32) :=
  let afterO := (TileShape.allIndices [c.BTL,c.BV]).foldl (fun t i =>
    let row := queryChunk c s+i.1.val
    let col := valueChunk c s+i.2.1.val
    if row<c.T ∧ col<c.V then t.write c.o
      ((s.pids 2+c.B*c.H*(s.pids 0/((c.V+c.BV-1)/c.BV)))*c.s_vo_h+row*c.s_vo_t+col*c.s_vo_d)
      (.mk .real (values.output i)) else t) s
  (TileShape.allIndices [c.BTL]).foldl (fun t i =>
    let row := queryChunk c s+i.1.val
    if row<c.T then t.write c.z
      ((s.pids 2+c.B*c.H*(s.pids 0/((c.V+c.BV-1)/c.BV)))*c.T+row)
      (.mk .real (values.normalizer i)) else t) afterO

theorem forward_update_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) (diagonal : Bool) (start : Nat)
    (acc : ForwardAccumulator (FP.Value .fp32) c.BTL c.BV) (hscale : c.scale=1) :
    forwardUpdate (sequentialFP32 rest cfg) (softwareDot cfg) c s diagonal start acc =
      roundedForwardUpdate cfg c s diagonal start acc := by
  have hq : forwardQuery (sequentialFP32 rest cfg) c s = roundedQuery cfg c s := by
    funext i
    simp [forwardQuery, roundedQuery, matrixRead, wordMatrix, sequentialFP32,
      FP.Software.scalarFP32, FP.Software.fp32Context, hscale, one32]
  have hm (R : RegionName) (base rows cols rs cs ro co m n : Nat) :
      matrixRead (sequentialFP32 rest cfg) s R base rows cols rs cs ro co m n =
        wordMatrix s R base rows cols rs cs ro co m n := by rfl
  simp only [forwardUpdate, roundedForwardUpdate, hq, hm]
  norm_num [softwareDot, sequentialFP32, FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl

theorem forward_values_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) (hscale : c.scale=1) :
    forwardValues (sequentialFP32 rest cfg) (softwareDot cfg) c s = roundedForwardValues cfg c s := by
  have hu (diagonal : Bool) : forwardUpdate (sequentialFP32 rest cfg) (softwareDot cfg) c s diagonal =
      roundedForwardUpdate cfg c s diagonal := by
    funext start acc
    exact forward_update_software rest cfg c s diagonal start acc hscale
  simp only [forwardValues, roundedForwardValues, hu]
  simp [sequentialFP32, FP.Software.scalarFP32, FP.Software.fp32Context]

specification forward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) (hscale : c.scale=1) :
    ∃ final, FP.Structural.exec (withDot (sequentialFP32 rest cfg) (softwareDot cfg)) (forwardKernel c) s = some final ∧
      final.mem = (roundedOutputMemory c s (roundedForwardValues cfg c s)).mem := by
  obtain ⟨final,he,hm⟩ := forward_fp32 (sequentialFP32 rest cfg) (softwareDot cfg) c s
  refine ⟨final,he,?_⟩
  unfold forwardMemory at hm
  rw [forward_values_software rest cfg c s hscale] at hm
  simpa [roundedOutputMemory,
    sequentialFP32, FP.Software.scalarFP32, FP.Software.fp32Context] using hm

#axiomsClean forward_update_software
#axiomsClean forward_values_software
#axiomsClean forward_software_fp32
end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
