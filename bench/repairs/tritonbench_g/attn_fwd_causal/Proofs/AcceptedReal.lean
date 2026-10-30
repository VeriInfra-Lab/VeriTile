import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.RealInvariant
namespace VeriTile.Bench.Repairs.AttnFwdCausal
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

namespace Real
/-- Independent logical pre-call attention formula. Its inputs are the original
FP32 storage values interpreted as reals; every numerical operation and cast in
the source is precision-erased. This is not a native accuracy statement. -/
def expected (p : Parameters) (μ : RegionName → Nat → ℝ) (b h n d : Nat) : ℝ :=
  RealMath.expected p.length
    (fun m e=>μ p.Q (b*p.q.batch+h*p.q.head+m*p.q.row+e*p.q.dim))
    (fun j e=>μ p.K (b*p.k.batch+h*p.k.head+e*p.k.dim+j*p.k.row))
    (fun j e=>μ p.V (b*p.v.batch+h*p.v.head+j*p.v.row+e*p.v.dim))
    (fun block=>μ p.QS ((b*p.heads+h)*((p.length+127)/128)+block))
    (fun block=>μ p.KS ((b*p.heads+h)*((p.length+63)/64)+block)) n d

theorem allocation_reads (p : Parameters) (abi : InputABI p) (before : State R)
    (alloc : Allocation p abi before) (μ : RegionName → Nat → ℝ) (h : Reads p before μ) :
    Reads p alloc.state μ := by
  intro r hr off
  have hf : r≠p.Out ∧ r≠alloc.unusedM.region := by
    have hq:=alloc.fresh.2 abi.query (by simp)
    have hk:=alloc.fresh.2 abi.key (by simp)
    have hv:=alloc.fresh.2 abi.value (by simp)
    have hqs:=alloc.fresh.2 abi.queryScale (by simp)
    have hks:=alloc.fresh.2 abi.keyScale (by simp)
    rw [abi.regions.1] at hq
    rw [abi.regions.2.1] at hk
    rw [abi.regions.2.2.1] at hv
    rw [abi.regions.2.2.2.1] at hqs
    rw [abi.regions.2.2.2.2] at hks
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl
    · exact ⟨Ne.symm hq.2.2.1,Ne.symm hq.2.2.2⟩
    · exact ⟨Ne.symm hk.2.2.1,Ne.symm hk.2.2.2⟩
    · exact ⟨Ne.symm hv.2.2.1,Ne.symm hv.2.2.2⟩
    · exact ⟨Ne.symm hqs.2.2.1,Ne.symm hqs.2.2.2⟩
    · exact ⟨Ne.symm hks.2.2.1,Ne.symm hks.2.2.2⟩
  rw [alloc.frame r off hf.1 hf.2]
  exact h r hr off

theorem expected_at (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (b h n d : Nat) (hh : h<p.heads) (hd : d<128) :
    RealMath.expected p.length
      (qAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (kAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (vAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (qsAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (ksAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      ((start (grid p) (n/128,b*p.heads+h,0) s).pids 0*128+(lane n d hd).1.val)
      (lane n d hd).2.1.val=expected p μ b h n d := by
  have hp : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hrow : n/128*128+n%128=n := by omega
  delta qAt kAt vAt qsAt ksAt
  simp [expected,base,start,ids,lane,hdiv,hmod,hrow]

/-- Public whole-call precision-erased correctness: actual current-memory
program execution, arbitrary legal CTA order, complete logical returned tensor,
and frame of every original input/all other live regions. -/
theorem correct (p : Parameters) (_guard : Guard p) (storage : TensorStorage p)
    (abi : InputABI p) (before : State R) (alloc : Allocation p abi before)
    (μ : RegionName → Nat → ℝ) (h : Reads p before μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi before alloc work=some result ∧
      result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out → r≠alloc.unusedM.region → result.state.mem r off=before.mem r off) := by
  obtain ⟨result,hr,ht,hv,hf⟩:=host_result (fun _=>algebra) (fun _=>dotModel) p storage abi before alloc work legal
  refine ⟨result,hr,ht,?_,hf⟩
  intro b hb hh hH n hn d hd
  rw [hv b hb hh hH n hn d hd]
  have hreads : Reads p (start (grid p) (n/128,b*p.heads+hh,0) alloc.state) μ :=
    allocation_reads p abi before alloc μ h
  have ha : active p (start (grid p) (n/128,b*p.heads+hh,0) alloc.state) (lane n d hd) :=
    ⟨by change n/128*128+n%128<p.length;omega,hd⟩
  rw [answer_real p _ μ hreads _ ha,expected_at p alloc.state μ b hh n d hH hd]

#specNonCircular expected avoiding [invoke,runPrograms,FP.Structural.exec,accumulated]
#axiomsClean correct
end Real
end
end VeriTile.Bench.Repairs.AttnFwdCausal
