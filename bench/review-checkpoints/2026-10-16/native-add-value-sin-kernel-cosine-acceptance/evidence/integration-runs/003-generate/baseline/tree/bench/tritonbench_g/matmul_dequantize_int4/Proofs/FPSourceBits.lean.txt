import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.SourceBits
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

/-- Closed unused fallback. The full-memory theorem proves that every choice
of fallback gives this same reference; no unsupported primitive is assumed. -/
def unused : Algebra Word where
  literal := fun _ _ _ => FP.Value.zero .fp32
  negInf := FP.Value.zero .fp32
  binary := fun _ _ _ _ _ => FP.Value.zero .fp32
  unary := fun _ _ _ => FP.Value.zero .fp32
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => FP.Value.zero .fp32
  fromInt := fun _ _ => FP.Value.zero .fp32
  fp32Bits := fun _ => FP.Value.zero .fp32
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32

/-- Independent per-block source-profile dot: starts at positive zero;
widens half operands, rounds each fp32 multiply, then rounds each fp32 add in
ascending K order. This is deliberately not a live-carry FMA schedule. -/
def blockBits (p : Parameters) (s : State Word) (step : Nat) (i : CIndex p) : Word :=
  (List.finRange p.blockInner).foldl (fun acc t =>
    FP.add .ieeeValue acc (FP.mul .ieeeValue
      (QuantizeKvTransform.MixedBits.widen (FPSoftware.machineA p s step (i.1,t,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.widen (FPSoftware.dequantized p s step (t,i.2.1,PUnit.unit)))))
    (FP.Value.zero .fp32)

/-- The original source then performs an additional rounded add between the
previous accumulator and this dot-from-zero result. All half dequantization
rounding points are kept by `dequantized`; no algebraic factoring is used. -/
def sourceAccum (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step+1 => fun i => FP.add .ieeeValue (sourceAccum p s step i) (blockBits p s step i)

def sourceOutput (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (sourceAccum p s (steps p) i)

def sourceReference (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then
      t.write p.output (outputOffset p s i) (.mk .fp16 (sourceOutput p s i)) else t) s

theorem machineAccum_eq (p : Parameters) (s : State Word) (step : Nat) :
    FPSoftware.machineAccum FPSoftware.softwareDot p s step = sourceAccum p s step := by
  induction step with
  | zero => rfl
  | succ step ih =>
    simp only [FPSoftware.machineAccum,sourceAccum,ih]
    simp only [FPSoftware.softwareDot,Int4Matmul.FPSoftware.dot,blockBits]

theorem output_eq (p : Parameters) (s : State Word) :
    FPSoftware.machineOutput FPSoftware.softwareDot p s = sourceOutput p s := by
  funext i
  simp only [FPSoftware.machineOutput,sourceOutput,machineAccum_eq]

#axiomsClean machineAccum_eq
#axiomsClean output_eq
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.SourceBits
