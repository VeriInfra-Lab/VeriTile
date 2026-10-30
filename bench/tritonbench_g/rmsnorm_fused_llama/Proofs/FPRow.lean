import bench.tritonbench_g.rmsnorm_fused_llama.Kernels

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Row
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def lit (A : Algebra α) (x : ℝ) := A.literal (some .fp32) .real x
def add (A : Algebra α) := A.binary (some .fp32) .real .add
def mul (A : Algebra α) := A.binary (some .fp32) .real .mul
def div (A : Algebra α) := A.binary (some .fp32) .real .div
def cast (A : Algebra α) := A.cast (some .fp32) .real .real
def read (s : State α) (r : RegionName) (o : Nat) := (s.mem r o).read .real
def xload (A : Algebra α) (s : State α) (X : RegionName) (row N k : Nat) :=
  cast A (if k<N then read s X (row*N+k) else lit A 0)
def wload (A : Algebra α) (s : State α) (W : RegionName) (N k : Nat) :=
  cast A (if k<N then read s W k else s.undef .real W k)
def square (A : Algebra α) (s : State α) (X : RegionName) (row N B : Nat) : Values α .real [B] :=
  fun i => add A (lit A 0) (mul A (xload A s X row N i.1.val) (xload A s X row N i.1.val))
def variance (A : Algebra α) (s : State α) (X : RegionName) (row N B : Nat) :=
  div A (A.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
    (square A s X row N B) PUnit.unit) (lit A (N : ℝ))
def rstd (A : Algebra α) (s : State α) (X : RegionName) (row N B : Nat) (eps : ℝ) :=
  div A (lit A 1) (A.unary (some .fp32) .sqrt (add A (variance A s X row N B) (lit A eps)))
def value (A : Algebra α) (s : State α) (X W : RegionName) (row N B : Nat) (eps : ℝ) (k : Nat) :=
  A.cast (some .fp32) .fp16 .real (A.cast (some .fp32) .real .fp16
    (mul A (mul A (xload A s X row N k) (rstd A s X row N B eps)) (wload A s W N k)))
def block (A : Algebra α) (s : State α) (X Y W : RegionName) (row N B : Nat) (eps : ℝ) :=
  (TileShape.allIndices [B]).foldl
    (fun st i => if i.1.val<N then st.write Y (row*N+i.1.val) (.mk .real (value A s X W row N B eps i.1.val)) else st) s

theorem write_congr {ι : Type} (xs : List ι) (Y : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Cell α)
    (s t : State α) (h : s.mem=t.mem) :
    (xs.foldl (fun st i => if active i then st.write Y (off i) (v i) else st) s).mem =
      (xs.foldl (fun st i => if active i then st.write Y (off i) (v i) else st) t).mem := by
  induction xs generalizing s t with
  | nil => exact h
  | cons i xs ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all [State.write]

/-- The complete unchanged two-loop source, in the wrapper's actual FP32
single-tile regime, executes and emits this typed scalar graph. -/
theorem complete (A : Algebra α) (s : State α) (X Y W : RegionName)
    (N B : Nat) (eps : ℝ) (hN : 0<N) (hNB : N≤B) :
    ∃ t, exec A (kernel X Y W N B eps) s=some t ∧
      t.mem=(block A s X Y W (s.pids 0) N B eps).mem := by
  have hB : B≠0 := by omega
  have hz : (0.0 : ℝ)=0 := by norm_num
  simp [kernel,VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.originalTyped,
    FP.Structural.exec,FP.Structural.run,FP.Structural.step,FP.Structural.range,evalExpr,evalComputeOp,evalOp_unfold,store,
    numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,State.setReg,Option.bind,hN,hB,hz,Nat.not_lt.mpr hNB,
    block,value,rstd,variance,square,xload,wload,lit,add,mul,Row.div,cast,read]
  apply write_congr
  rfl

#axiomsClean complete
#specNonCircular value avoiding [kernel,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Row
