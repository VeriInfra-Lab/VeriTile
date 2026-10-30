import bench.tritonbench_g.llama_ff_triton.Proofs.RealArithmetic
import bench.tritonbench_g.llama_ff_triton.Proofs.Logical

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Real
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def parameters (p : Execution.Parameters) : LlamaFF.Parameters ℝ := ⟨p.inner,p.epsilon⟩
def readWeight (p : Execution.Parameters) (s : State ℝ) (r : RegionName) (offset : Nat) : ℝ :=
  if p.byte then decodeReal (Float8E5Bits.ofInt ((s.mem r offset).read .int))
  else (s.mem r offset).read .fp16
def inputs (p : Execution.Parameters) (s : State ℝ) (row : Nat) (weight : RegionName) : LlamaFF.RowInput ℝ where
  x := fun k => (s.mem "A" (row*p.inner+k)).read .fp16
  rms := readWeight p s "RMS"
  weight := fun n k => readWeight p s weight (k+n*p.inner)
  weight3 := fun n k => readWeight p s "W3" (k+n*p.inner)

theorem rowX_masked (p : Execution.Parameters) (s : State ℝ) (row t : Nat) (k : Fin 64) (r : RegionName) :
    rowX realArithmetic p s row t k=
      LlamaFF.masked LlamaFF.realOps (parameters p) (inputs p s row r).x t k := by
  rfl
theorem rms_masked (p : Execution.Parameters) (s : State ℝ) (row t : Nat) (k : Fin 64) (r : RegionName) :
    rmsValue realArithmetic p s t k.val=
      LlamaFF.masked LlamaFF.realOps (parameters p) (inputs p s row r).rms t k := by
  by_cases h : t*64+k.val<p.inner
  · simp [rmsValue,rmsByte,LlamaFF.masked,parameters,inputs,activeK,h,readWeight,realArithmetic]
  · simp [rmsValue,rmsByte,LlamaFF.masked,parameters,inputs,activeK,h,readWeight,realArithmetic,
      halfZero,LlamaFF.realOps,decodeReal_zero]
theorem rowW_masked (p : Execution.Parameters) (s : State ℝ) (row n t : Nat) (k : Fin 64) (r : RegionName) :
    rowW realArithmetic p s r n t k=
      LlamaFF.masked LlamaFF.realOps (parameters p) ((inputs p s row r).weight n) t k := by
  by_cases h : t*64+k.val<p.inner
  · simp [rowW,LlamaFF.masked,parameters,inputs,h,readWeight,realArithmetic]
  · simp [rowW,LlamaFF.masked,parameters,inputs,h,readWeight,realArithmetic,
      halfZero,LlamaFF.realOps,decodeReal_zero]
theorem scaled_masked (p : Execution.Parameters) (s : State ℝ) (row t : Nat) (k : Fin 64) (r : RegionName) :
    rowScaled realArithmetic p s row t k=
      LlamaFF.scaled LlamaFF.realOps (parameters p) (inputs p s row r) t k := by
  simp only [rowScaled,rowX_masked p s row t k r,rms_masked p s row t k r]
  rfl
theorem chain_dot (p : Execution.Parameters) (s : State ℝ) (row n t : Nat) (r : RegionName) (old : ℝ) :
    chain realArithmetic (rowScaled realArithmetic p s row t) (rowW realArithmetic p s r n t) old=
      LlamaFF.dotBlock LlamaFF.realOps (parameters p) (inputs p s row r) LlamaFF.orderedK16 n t old := by
  rw [chain64_ordered]
  simp only [LlamaFF.dotBlock,LlamaFF.instruction,scaled_masked p s row t _ r,rowW_masked p s row n t _ r]
  rfl

theorem accumulation (p : Execution.Parameters) (s : State ℝ) (row n : Nat) (r : RegionName) (t : Nat) :
    rowAcc realArithmetic Bool.false p s r row n t=
      LlamaFF.accumulation LlamaFF.realOps (parameters p) (inputs p s row r) (fun _ => LlamaFF.orderedK16) n t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change rowAcc realArithmetic Bool.false p s r row n t+
      chain realArithmetic (rowScaled realArithmetic p s row t) (rowW realArithmetic p s r n t) 0=
      LlamaFF.dotBlock LlamaFF.realOps (parameters p) (inputs p s row r) LlamaFF.orderedK16 n t
        (LlamaFF.accumulation LlamaFF.realOps (parameters p) (inputs p s row r) (fun _ => LlamaFF.orderedK16) n t)
    rw [ih,chain_dot,LlamaFF.dotBlock_real,LlamaFF.dotBlock_real]
    ring
theorem moment (p : Execution.Parameters) (s : State ℝ) (row : Nat) (r : RegionName) (k : Fin 64) (t : Nat) :
    rowMoment realArithmetic p s row k t=
      LlamaFF.moment LlamaFF.realOps (parameters p) (inputs p s row r) k t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change rowMoment realArithmetic p s row k t+(rowX realArithmetic p s row t k)^2=
      LlamaFF.moment LlamaFF.realOps (parameters p) (inputs p s row r) k t+
        1*(LlamaFF.masked LlamaFF.realOps (parameters p) (inputs p s row r).x t k*
          LlamaFF.masked LlamaFF.realOps (parameters p) (inputs p s row r).x t k)
    rw [ih,rowX_masked p s row t k r]
    ring
theorem norm (p : Execution.Parameters) (s : State ℝ) (row : Nat) (r : RegionName) :
    rowNorm realArithmetic p s row=
      LlamaFF.realNorm (parameters p) (inputs p s row r) := by
  rw [←LlamaFF.norm_real _ _ LlamaFF.butterfly64]
  have hr : reduceLine realArithmetic (some .fp32) (fun k => rowMoment realArithmetic p s row k (steps p))=
      LlamaFF.butterfly64.tree.eval (·+·)
        (fun k => LlamaFF.moment LlamaFF.realOps (parameters p) (inputs p s row r) k (LlamaFF.steps p.inner)) := by
    change LlamaFF.butterfly64.tree.eval (·+·) _=_
    congr 1
    funext k
    exact moment p s row r k (steps p)
  change 1/Real.sqrt ((reduceLine realArithmetic (some .fp32)
    (fun k => rowMoment realArithmetic p s row k (steps p)))/(p.inner:ℝ)+p.epsilon)=_
  rw [hr]
  rfl

/-- Ordinary decoded RMSNorm–SwiGLU formula: finite E5M2 values are exact,
and the explicitly chosen ideal interpretation maps exceptional bytes to zero. -/
def target (p : Execution.Parameters) (s : State ℝ) (row column : Nat) : ℝ :=
  LlamaFF.realTarget (parameters p) (inputs p s row "W1") column

theorem source_formula (p : Execution.Parameters) (s : State ℝ) (row column : Nat) :
    logicalSource realArithmetic p s row column=target p s row column := by
  simp only [logicalSource,accumulation,norm p s row "W1"]
  rw [show steps p=LlamaFF.steps (parameters p).K from rfl]
  rw [LlamaFF.projection_sum,LlamaFF.projection_sum]
  simp only [target,LlamaFF.realTarget,LlamaFF.realNorm,inputs,realArithmetic]

#axiomsClean accumulation
#axiomsClean moment
#axiomsClean norm
#axiomsClean source_formula
#specNonCircular target avoiding [source,run,logicalSource,rowAcc,rowMoment]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Real
