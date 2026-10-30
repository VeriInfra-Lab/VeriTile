import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Host
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Ideal

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent pointwise producer reference: expected local scalar values at
each output address and the original memory everywhere else. No execution or
native result appears in its definition. -/
def quantMemoryReference (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (m : Memory F H) (p : Address) : Cell F H :=
  if h : ∃ x : QuantIndex G K,p=q.writeAddress x then quantExpected a q plan m h.choose else m p
theorem quantWrapper_reference (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (trials : List (QuantInvocation G K)) (selected : QuantInvocation G K) (m : Memory F H) :
    quantWrapper a q trials selected m=quantMemoryReference a q selected.plan m := by
  funext p
  unfold quantMemoryReference
  split_ifs with h
  · exact (congrArg (quantWrapper a q trials selected m) h.choose_spec).trans
      (quantWrapper_at a q sep trials selected m h.choose)
  · apply quantWrapper_frame a q sep
    intro x equal
    exact h ⟨x,equal⟩

def quantScaleTarget (value : Nat → ℝ) (K : Nat) : ℝ := maximumAbs value K/127
def quantValueTarget (value : Nat → ℝ) (K k : Nat) : Byte := realInteger8 (value k/quantScaleTarget value K)
theorem Reduction.eval_real (tree : Reduction K) (value : Fin K → ℝ) :
    tree.eval realOps value=tree.real value := by
  induction tree with
  | zero => rfl
  | lane k => rfl
  | maximum left right hl hr =>
    change max (left.eval realOps value) (right.eval realOps value)=max (left.real value) (right.real value)
    rw [hl,hr]
theorem quantScale_real (q : QuantBuffers) (plan : Fin G → ReductionPlan K) (m : Memory ℝ ℝ) (g : Fin G) :
    quantScale realOps q plan m g=quantScaleTarget (fun k => read32 realOps.toOps m (q.inputAddress g k)) K := by
  unfold quantScale
  rw [Reduction.eval_real,(plan g).real_target (fun k => read32 realOps.toOps m (q.inputAddress g k))]
  have nonnegative := maximumAbs_nonnegative (fun k => read32 realOps.toOps m (q.inputAddress g k)) K
  cases mode : q.initialMaximumZero <;> simp [mode,realOps,quantScaleTarget,max_eq_right nonnegative]
  exact nonnegative
theorem quantValue_real (q : QuantBuffers) (plan : Fin G → ReductionPlan K) (m : Memory ℝ ℝ) (x : QuantCell G K) :
    quantValue realOps q plan m x=quantValueTarget (fun k => read32 realOps.toOps m (q.inputAddress x.1 k)) K x.2 := by
  unfold quantValue
  rw [quantScale_real]
  rfl

def composedReference (a : Ops I F H) (s : Shape) (q : QuantBuffers) (b : Buffers)
    (qplan : Fin s.M → ReductionPlan s.K) (chosen : Invocation s)
    (m : Memory F H) (x : MatrixCell s) : Stored F H :=
  wordReference a s chosen.config b chosen.selected chosen.schedule (quantMemoryReference a q qplan m) x
theorem composed_reference (a : Ops I F H) (s : Shape) (q : QuantBuffers) (qsep : q.Separate)
    (b : Buffers) (sep : b.Separate) (qtrials : List (QuantInvocation s.M s.K))
    (qchosen : QuantInvocation s.M s.K) (owner callerDevice : Nat)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory F H) (x : MatrixCell s) :
    (composed a s q b qtrials qchosen owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      .inr (composedReference a s q b qchosen.plan chosen m x) := by
  rw [composed_word a s q b sep,quantWrapper_reference a q qsep]
  rfl
#specNonCircular quantMemoryReference avoiding [quantRun,quantWrapper,quantGridCode,Code.native]
#specNonCircular composedReference avoiding [composed,wrapper,gridRun,Code.native]
#specNonCircular quantScaleTarget avoiding [Reduction.eval,Reduction.code]
#axiomsClean quantWrapper_reference
#axiomsClean quantScale_real
#axiomsClean quantValue_real
#axiomsClean composed_reference
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
