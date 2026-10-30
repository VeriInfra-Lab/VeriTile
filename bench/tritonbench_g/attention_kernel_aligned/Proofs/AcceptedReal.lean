import bench.tritonbench_g.attention_kernel_aligned.Proofs.RealInvariant
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Real
open VeriTile Triton FP.Structural Source Whole Full
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Independent pre-call base-2 bias attention. Arithmetic and casts are erased
into Real operations; this statement makes no native accuracy claim. -/
def expected (p : Parameters) (μ : RegionName → Nat → ℝ) (b h n d : Nat) : ℝ :=
  RealMath.expected p.length p.head p.blockN p.biasLast p.scale
    (fun m e=>μ p.q (b*p.qBatchStride+h*p.qHeadStride+m*p.qRowStride+e*p.qColStride))
    (fun j e=>μ p.k (b*p.kBatchStride+h*p.kHeadStride+e*p.kDimStride+j*p.kSeqStride))
    (fun j e=>μ p.v (b*p.vBatchStride+h*p.vHeadStride+j*p.vSeqStride+e*p.vDimStride))
    (fun m j=>μ p.bias (b*p.biasBatchStride+h*p.biasHeadStride+m*p.biasRowStride+j)) n d

theorem expected_at (p : Parameters) (selected : Selected p) (s : State R) (μ : RegionName → Nat → ℝ)
    (b h n d : Nat) (hh : h<p.heads) (hd : d<p.head) :
    RealMath.expected p.length p.head p.blockN p.biasLast p.scale
      (qAt p (start (grid p) (n/p.blockM,b*p.heads+h,0) s) μ)
      (kAt p (start (grid p) (n/p.blockM,b*p.heads+h,0) s) μ)
      (vAt p (start (grid p) (n/p.blockM,b*p.heads+h,0) s) μ)
      (biasAt p (start (grid p) (n/p.blockM,b*p.heads+h,0) s) μ)
      (row p (start (grid p) (n/p.blockM,b*p.heads+h,0) s) (lane p selected n d hd).1)
      (lane p selected n d hd).2.1.val=expected p μ b h n d := by
  have hp : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hrow : n/p.blockM*p.blockM+n%p.blockM=n := by have:=Nat.div_add_mod' n p.blockM; omega
  delta qAt kAt vAt biasAt
  simp [expected,headBase,kvBase,valueBase,biasBase,row,start,ids,lane,hdiv,hmod,hrow]

theorem correct (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State R) (μ : RegionName → Nat → ℝ) (h : Reads p before μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>model) p abi before work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        result.state.mem p.output (logicalAddress p b h n d)=.mk .fp16 (some (expected p μ b h n d))) ∧
      (∀ r off,OutsideLogical p r off → result.state.mem r off=before.mem r off) := by
  obtain ⟨result,hr,ht,hnone,hv,hf⟩:=host_result (fun _=>algebra) (fun _=>model) p guard storage abi before work legal
  refine ⟨result,hr,ht,hnone,?_,hf⟩
  intro b hb hh hH n hn d hd
  rw [hv b hb hh hH n hn d hd]
  have hreads : Reads p (start (grid p) (n/p.blockM,b*p.heads+hh,0) before) μ := h
  have ha : row p (start (grid p) (n/p.blockM,b*p.heads+hh,0) before) (lane p guard.selected n d hd).1<p.length := by
    change n/p.blockM*p.blockM+n%p.blockM<p.length
    have:=Nat.div_add_mod' n p.blockM
    omega
  change Cell.mk .fp16 (outputValue model p _ _)=_
  rw [answer_real p _ μ hreads _ ha (guard.selected.blockN_pos p),expected_at p guard.selected before μ b hh n d hH hd]

#specNonCircular expected avoiding [invoke,runPrograms,FP.Structural.exec,rounds]
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Real
