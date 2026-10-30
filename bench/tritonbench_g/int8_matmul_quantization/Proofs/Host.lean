import bench.tritonbench_g.int8_matmul_quantization.Proofs.Frame
import bench.tritonbench_g.int8_matmul_quantization.Proofs.Producer
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Host

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000
abbrev Tensor := Int8DequantMatmul.Tensor
abbrev DType := Int8DequantMatmul.DType
def precisionDType : Precision → DType | .half => .fp16 | .single => .fp32
def outputTensor (s : Shape) (b : Buffers) (device : Nat) : Tensor :=
  ⟨b.out,b.outOffset,[s.M,s.N],[max s.N 1,1],precisionDType b.outPrecision,device⟩
structure Invocation (s : Shape) where
  config : Config
  selected : Selection s config
  schedule : Schedule s config
def invoke (a : Ops I F H) (s : Shape) (b : Buffers) (v : Invocation s) (m : Memory F H) :=
  launch a s v.config b v.selected v.schedule m
def trials (a : Ops I F H) (s : Shape) (b : Buffers) : List (Invocation s) → Memory F H → Memory F H
  | [],m => m
  | v::vs,m => trials a s b vs (invoke a s b v m)
theorem trials_frame (a : Ops I F H) (s : Shape) (b : Buffers) (vs : List (Invocation s)) (m : Memory F H) :
    InputsAgree b (trials a s b vs m) m := by
  intro p hp
  induction vs generalizing m with
  | nil => rfl
  | cons v vs ih => rw [trials,ih]; exact launch_frame _ _ _ _ _ _ _ p hp
theorem trials_full_frame (a : Ops I F H) (s : Shape) (b : Buffers)
    (vs : List (Invocation s)) (m : Memory F H) (p : Address) (outside : ¬OutputRegion s b p) :
    trials a s b vs m p=m p := by
  induction vs generalizing m with
  | nil => rfl
  | cons v vs ih => rw [trials,ih]; exact launch_full_frame _ _ _ _ _ _ _ p outside
theorem reference_current (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (sep : b.Separate)
    (selected : Selection s c) (q : Schedule s c) (m before : Memory F H)
    (same : InputsAgree b m before) (x : MatrixCell s) :
    wordReference a s c b selected q m x=wordReference a s c b selected q before x := by
  unfold wordReference
  congr 1
  funext z
  exact partial_current a s c b sep selected m before same _ _ _

/-- Literal successful Autotuner lifecycle: wrapper zeros once; every measured
trial resets; a cold run resets after benchmarking; a cache hit has no such
post-benchmark hook. The final kernel itself does not invent another hook. -/
def runtime (a : Ops I F H) (s : Shape) (b : Buffers) (benchmarks : List (Invocation s))
    (chosen : Invocation s) (m : Memory F H) : Memory F H :=
  let initial := reset a s b m
  let done := trials a s b benchmarks initial
  let ready := if benchmarks=[] then done else reset a s b done
  gridRun a s chosen.config b chosen.selected chosen.schedule.events ready
theorem runtime_normalized (a : Ops I F H) (s : Shape) (b : Buffers)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory F H) :
    runtime a s b benchmarks chosen m=invoke a s b chosen (trials a s b benchmarks m) := by
  cases benchmarks with
  | nil => rfl
  | cons v vs => simp [runtime,trials,invoke,launch,reset_idempotent]

structure Returned (F H : Type) where
  tensor : Tensor
  memory : Memory F H
  currentDevice : Nat
def wrapper (a : Ops I F H) (s : Shape) (b : Buffers) (owner callerDevice : Nat)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory F H) : Returned F H :=
  if s.M=0 ∨ s.N=0 then ⟨outputTensor s b owner,reset a s b m,callerDevice⟩
  else
    let invoked := DequantizeMatmul.deviceCall owner
      (fun entered => ⟨entered.currentDevice,runtime a s b benchmarks chosen entered.payload⟩)
      ⟨callerDevice,m⟩
    ⟨outputTensor s b owner,invoked.payload,invoked.currentDevice⟩
theorem wrapper_returns (a : Ops I F H) (s : Shape) (b : Buffers) (owner callerDevice : Nat)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory F H) :
    (wrapper a s b owner callerDevice benchmarks chosen m).tensor=outputTensor s b owner ∧
    (wrapper a s b owner callerDevice benchmarks chosen m).currentDevice=callerDevice := by
  unfold wrapper
  split_ifs <;> exact ⟨rfl,rfl⟩
theorem wrapper_word (a : Ops I F H) (s : Shape) (b : Buffers) (sep : b.Separate)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (m : Memory F H) (x : MatrixCell s) :
    (wrapper a s b owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      .inr (wordReference a s chosen.config b chosen.selected chosen.schedule m x) := by
  have nonempty : ¬(s.M=0 ∨ s.N=0) := by have hm:=x.1.isLt; have hn:=x.2.isLt; omega
  simp only [wrapper,if_neg nonempty,DequantizeMatmul.deviceCall,runtime_normalized,invoke]
  rw [launch_word a s chosen.config b sep]
  rw [reference_current a s chosen.config b sep _ _ _ m (trials_frame a s b benchmarks m)]
theorem wrapper_full_frame (a : Ops I F H) (s : Shape) (b : Buffers)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (m : Memory F H) (p : Address) (outside : ¬OutputRegion s b p) :
    (wrapper a s b owner callerDevice benchmarks chosen m).memory p=m p := by
  unfold wrapper
  split_ifs
  · exact reset_full_frame _ _ _ _ _ outside
  · simp only [DequantizeMatmul.deviceCall,runtime_normalized,invoke]
    rw [launch_full_frame _ _ _ _ _ _ _ _ outside,trials_full_frame _ _ _ _ _ _ outside]

/-- The row producer's actual final memory is passed to GEMM. This definition
has no independently supplied quantized-input oracle. -/
def composed (a : Ops I F H) (s : Shape) (q : QuantBuffers) (b : Buffers)
    (qtrials : List (QuantInvocation s.M s.K)) (qchosen : QuantInvocation s.M s.K)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (m : Memory F H) : Returned F H :=
  wrapper a s b owner callerDevice benchmarks chosen (quantWrapper a q qtrials qchosen m)
theorem composed_word (a : Ops I F H) (s : Shape) (q : QuantBuffers) (b : Buffers) (sep : b.Separate)
    (qtrials : List (QuantInvocation s.M s.K)) (qchosen : QuantInvocation s.M s.K)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (m : Memory F H) (x : MatrixCell s) :
    (composed a s q b qtrials qchosen owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      .inr (wordReference a s chosen.config b chosen.selected chosen.schedule (quantWrapper a q qtrials qchosen m) x) :=
  wrapper_word a s b sep owner callerDevice benchmarks chosen _ x

#axiomsClean runtime_normalized
#axiomsClean wrapper_returns
#axiomsClean wrapper_word
#axiomsClean wrapper_full_frame
#axiomsClean composed_word
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
