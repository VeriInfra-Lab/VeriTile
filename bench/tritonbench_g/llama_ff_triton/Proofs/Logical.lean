import bench.tritonbench_g.llama_ff_triton.Proofs.NativeEpilogue

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
open scoped Classical
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def rowX {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (row step : Nat) (lane : Fin 64) : α :=
  if step*64+lane.val<p.inner then (s.mem "A" (row*p.inner+(step*64+lane.val))).read .fp16
  else halfZero a
def rowW {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (weight : RegionName) (column step : Nat) (lane : Fin 64) : α :=
  if p.byte then a.cast none .real .fp16 (a.cast (some .fp32) .f8e5 .real
    (a.decode .f8e5 (Float8E5Bits.ofInt
      (if step*64+lane.val<p.inner then
        (s.mem weight (step*64+lane.val+column*p.inner)).read .int else 0))))
  else if step*64+lane.val<p.inner then
    (s.mem weight (step*64+lane.val+column*p.inner)).read .fp16 else halfZero a
def nativeRowW {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (weight : RegionName) (column step : Nat) (lane : Fin 64) : α :=
  if p.byte then a.cast none .f8e5 .fp16
    (a.decode .f8e5 (Float8E5Bits.ofInt
      (if step*64+lane.val<p.inner then
        (s.mem weight (step*64+lane.val+column*p.inner)).read .int else 0)))
  else if step*64+lane.val<p.inner then
    (s.mem weight (step*64+lane.val+column*p.inner)).read .fp16 else halfZero a
def rowScaled {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (row step : Nat) (lane : Fin 64) : α :=
  a.binary none .fp16 .mul (rowX a p s row step lane) (rmsValue a p s step lane.val)
def rowAcc {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State α) (weight : RegionName) (row column : Nat) : Nat → α
  | 0 => a.literal (some .fp32) .real 0
  | t+1 =>
    if native then chain a (rowScaled a p s row t) (nativeRowW a p s weight column t)
      (rowAcc a native p s weight row column t)
    else a.binary (some .fp32) .real .add (rowAcc a native p s weight row column t)
      (a.cast (some .fp32) .fp32 .real (chain a (rowScaled a p s row t)
        (rowW a p s weight column t) (a.literal (some .fp32) .real 0)))
def rowMoment {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (row : Nat) (lane : Fin 64) : Nat → α
  | 0 => a.literal (some .fp32) .real 0
  | t+1 => a.binary (some .fp32) .real .add (rowMoment a p s row lane t)
      (a.powi (some .fp32) (a.cast (some .fp32) .fp16 .real (rowX a p s row t lane)) 2)

theorem accumulation_row {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State α) (weight : RegionName) (t : Nat) (i : TileIndex [16,16]) :
    accumulation a native p s weight t i=
      rowAcc a native p s weight (inputRow p s i.1) (inputColumn p s i.2.1) t := by
  have hx (step : Nat) : (fun lane => scaledValue a p s step (i.1,lane,PUnit.unit))=
      rowScaled a p s (inputRow p s i.1) step := by
    funext lane
    simp [scaledValue,rowScaled,inputValue,rowX,activeK,aOffset]
  have hw (step : Nat) : (fun lane => weightValue a p s weight step (lane,i.2.1,PUnit.unit))=
      rowW a p s weight (inputColumn p s i.2.1) step := by
    funext lane
    simp [weightValue,rowW,weightByte,activeK,wOffset]
  have hwn (step : Nat) : (fun lane => nativeWeightValue a p s weight step (lane,i.2.1,PUnit.unit))=
      nativeRowW a p s weight (inputColumn p s i.2.1) step := by
    funext lane
    simp [nativeWeightValue,nativeRowW,weightByte,activeK,wOffset]
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [accumulation,tensor,rowAcc,ih,hx,hw,hwn]

theorem moment_row {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (t : Nat) (i : TileIndex [16,64]) :
    moment a p s t i=rowMoment a p s (inputRow p s i.1) i.2.1 t := by
  induction t with
  | zero => rfl
  | succ t ih => simp [moment,rowMoment,ih,inputValue,rowX,activeK,aOffset]

def rowNorm {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (row : Nat) : α :=
  a.unary (some .fp32) .rsqrt (a.binary (some .fp32) .real .add
    (a.binary (some .fp32) .real .div
      (reduceLine a (some .fp32) (fun lane => rowMoment a p s row lane (steps p)))
      (a.fromInt (some .fp32) p.inner)) (a.literal (some .fp32) .real p.epsilon))
theorem normalization_row {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (i : Fin 16) :
    normalization a p s i=rowNorm a p s (inputRow p s i) := by
  simp only [normalization,rowNorm,moment_row]

def logicalSource {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State α) (row column : Nat) : α :=
  let rho := rowNorm a p s row
  let u := a.binary (some .fp32) .real .mul (rowAcc a Bool.false p s "W1" row column (steps p)) rho
  let v := a.binary (some .fp32) .real .mul (rowAcc a Bool.false p s "W3" row column (steps p)) rho
  a.cast none .real .fp16 (a.binary (some .fp32) .real .mul
    (a.binary (some .fp32) .real .mul u (a.unary (some .fp32) .sigmoid u)) v)
def logicalNative {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (s : State α) (row column : Nat) : α :=
  let rho := rowNorm a p s row
  let raw1 := rowAcc a Bool.true p s "W1" row column (steps p)
  let raw3 := rowAcc a Bool.true p s "W3" row column (steps p)
  let mul := a.binary (some .fp32) .real .mul
  let u := mul raw1 rho
  let v := mul raw3 rho
  a.cast none .real .fp16 (mul v (mul u
    (g.sigmoidNeg (g.fma (g.neg raw1) rho (a.literal (some .fp32) .real 0)))))

theorem sourceOutput_logical {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (i : TileIndex [16,16]) :
    sourceOutput a p s i=logicalSource a p s (inputRow p s i.1) (inputColumn p s i.2.1) := by
  simp only [sourceOutput,projection,logicalSource,accumulation_row,normalization_row]
theorem nativeOutput_logical {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (s : State α) (i : TileIndex [16,16]) :
    nativeOutput a g p s i=logicalNative a g p s (inputRow p s i.1) (inputColumn p s i.2.1) := by
  simp only [nativeOutput,nativeValues,gateValue,logicalNative,accumulation_row,normalization_row]

#axiomsClean accumulation_row
#axiomsClean moment_row
#axiomsClean sourceOutput_logical
#axiomsClean nativeOutput_logical
#specNonCircular logicalSource avoiding [source,body,run,sourceOutput,accumulation,moment]
#specNonCircular logicalNative avoiding [source,body,run,nativeExecution,nativeOutput,gateValue,accumulation,moment]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
