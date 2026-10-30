import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.Physical
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.RealInvariant
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.RealReturned
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000


/-- Independent logical attention over exactly N keys and 96 features.
Inputs come from the initial memory and descriptor maps, never source execution. -/
def target (p : Parameters) (μ : RegionName → Nat → ℝ) (b h n d : Nat) : ℝ :=
  RealMath.expected p.length
    (fun m e=>μ p.Q (offset p.q b h m e))
    (fun j e=>μ p.K (offset p.k b h j e))
    (fun j e=>μ p.V (offset p.v b h j e))
    (fun block=>μ p.QS ((b*p.heads+h)*((p.length+127)/128)+block))
    (fun block=>μ p.KS ((b*p.heads+h)*((p.length+63)/64)+block)) n d

/-- Concrete finite initial Real inputs and their independent total logical
extension make Reads realizable; no infinite-storage or output premise is used. -/
structure InputValues where
  capacity : RegionName → Nat
  values : (r : RegionName) → Fin (capacity r) → ℝ

def InputValues.store (m : InputValues) : Physical.Store Real.R :=
  ⟨m.capacity,(fun r i=>.mk .real (some (m.values r i))),.mk .real (some 0)⟩

def InputValues.logical (m : InputValues) (r : RegionName) (off : Nat) : ℝ :=
  if h : off<m.capacity r then m.values r ⟨off,h⟩ else 0

theorem input_values_reads (p : Parameters) (m : InputValues) :
    Real.Reads p m.store.extend m.logical := by
  intro r _ off
  by_cases h : off<m.capacity r
  · simp [InputValues.store,Physical.Store.extend,InputValues.logical,h]
  · simp [InputValues.store,Physical.Store.extend,InputValues.logical,h]

theorem allocation_input_frame {α : Type} [Inhabited α] (p : Parameters)
    (abi : InputABI p) (before : State α) (alloc : Allocation p abi before) :
    ∀ r∈[p.Q,p.K,p.V,p.QS,p.KS],∀ off,alloc.state.mem r off=before.mem r off := by
  have fresh : ∀ r∈[p.Q,p.K,p.V,p.QS,p.KS],p.Out≠r ∧ alloc.unusedM.region≠r := by
    intro r hr
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl
    · simpa only [abi.regions.1] using (alloc.fresh.2 abi.query (by simp)).2.2
    · simpa only [abi.regions.2.1] using (alloc.fresh.2 abi.key (by simp)).2.2
    · simpa only [abi.regions.2.2.1] using (alloc.fresh.2 abi.value (by simp)).2.2
    · simpa only [abi.regions.2.2.2.1] using (alloc.fresh.2 abi.queryScale (by simp)).2.2
    · simpa only [abi.regions.2.2.2.2] using (alloc.fresh.2 abi.keyScale (by simp)).2.2
  intro r hr off
  have h:=fresh r hr
  exact alloc.frame r off (Ne.symm h.1) (Ne.symm h.2)

theorem allocation_reads (p : Parameters) (abi : InputABI p) (before : State Real.R)
    (alloc : Allocation p abi before) (μ : RegionName → Nat → ℝ) (h : Real.Reads p before μ) :
    Real.Reads p alloc.state μ := by
  intro r hr off
  rw [allocation_input_frame p abi before alloc r hr off]
  exact h r hr off

theorem target_at_program (p : Parameters) (s : State Real.R) (μ : RegionName → Nat → ℝ)
    (b h n d : Nat) (hh : h<p.heads) (hd : d<96) :
    RealMath.expected p.length
      (Real.qAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (Real.kAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (Real.vAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (Real.qsAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      (Real.ksAt p (start (grid p) (n/128,b*p.heads+h,0) s) μ)
      ((start (grid p) (n/128,b*p.heads+h,0) s).pids 0*128+(lane n d hd).1.val)
      (lane n d hd).2.1.val = target p μ b h n d := by
  have hp : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hn : n/128*128+n%128=n := by omega
  unfold target Real.qAt Real.kAt Real.vAt Real.qsAt Real.ksAt
  simp [base,start,ids,lane,offset,hdiv,hmod,hn] <;>
    simp [Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

theorem returned_real (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : Physical.Store Real.R) (fits : Physical.InputFits p before)
    (alloc : Allocation p abi before.extend) (μ : RegionName → Nat → ℝ)
    (reads : Real.Reads p before.extend μ) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p abi before alloc work=some result ∧
      result.tensor=alloc.output ∧ result.memory.capacity p.Out=words p ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk .bf16 (some (target p μ b h n d))) ∧
      (∀ r (i : Fin (before.capacity r)),r≠p.Out → r≠alloc.unusedM.region →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=before.cells r i) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before := by
  obtain ⟨result,hr,ht,hcap,hv,hf,ha,hc,hinput⟩:=Physical.finite_return
    (fun _=>Real.algebra) (fun _=>Real.dotModel) p bases domain abi before fits alloc work legal
  refine ⟨result,hr,ht,hcap,?_,hf,ha,hc,hinput⟩
  intro b hb h hh n hn d hd
  obtain ⟨index,hi,hv⟩:=hv b hb h hh n hn d hd
  refine ⟨index,hi,?_⟩
  have hreads : Real.Reads p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) μ :=
    allocation_reads p abi before.extend alloc μ reads
  have active : Source.active p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd) := by
    change n/128*128+n%128<p.length ∧ d<96
    exact ⟨by omega,hd⟩
  rw [Real.answer_real p _ μ hreads _ active,target_at_program p alloc.state μ b h n d hh hd] at hv
  exact hv

#axiomsClean input_values_reads
#axiomsClean returned_real
#specNonCircular target avoiding [FP.Structural.exec,Whole.invoke,Physical.invoke,Whole.answer]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.RealReturned
