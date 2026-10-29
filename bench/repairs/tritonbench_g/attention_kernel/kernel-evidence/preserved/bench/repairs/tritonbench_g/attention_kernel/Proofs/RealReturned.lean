import bench.repairs.tritonbench_g.attention_kernel.Proofs.Physical
import bench.repairs.tritonbench_g.attention_kernel.Proofs.RealInvariant
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.RealReturned
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent logical attention over exactly N keys and D features, with
the source's two relative-bias windows and exact literal coefficient. -/
def target (p : Parameters) (μ : RegionName→ Nat→ℝ) (b h n d : Nat) : ℝ :=
  RealMath.expected p.length p.dim p.bn p.scale
    (fun m e=>μ p.Q (offset p.q b h m e))
    (fun j e=>μ p.K (offset p.k b h j e))
    (fun j e=>μ p.V (offset p.v b h j e))
    (fun m c=>μ p.Bias (offset p.bias b h m 0+c)) n d

structure InputValues where
  capacity : RegionName→ Nat
  values : (r : RegionName)→ Fin (capacity r)→ℝ

def InputValues.store (p : Parameters) (m : InputValues) : Physical.Store Real.R :=
  ⟨m.capacity,(fun r i=>.mk p.storage.tile (ofFloat p.storage.float (some (m.values r i)))),
    .mk p.storage.tile (ofFloat p.storage.float (some 0))⟩

def InputValues.logical (m : InputValues) (r : RegionName) (off : Nat) : ℝ :=
  if h : off<m.capacity r then m.values r ⟨off,h⟩ else 0

theorem input_values_reads (p : Parameters) (m : InputValues) :
    Real.Reads p (m.store p).extend m.logical := by
  intro r _ off
  by_cases h : off<m.capacity r <;>
    simp [InputValues.store,Physical.Store.extend,InputValues.logical,h]

theorem target_at_program (p : Parameters) (s : State Real.R) (μ : RegionName→ Nat→ℝ)
    (hbm : 0<p.bm) (b h n d : Nat) (hh : h<p.heads) (hd : d<p.dim) :
    RealMath.expected p.length p.dim p.bn p.scale
      (Real.qAt p (start (grid p) (n/p.bm,b*p.heads+h,0) s) μ)
      (Real.kAt p (start (grid p) (n/p.bm,b*p.heads+h,0) s) μ)
      (Real.vAt p (start (grid p) (n/p.bm,b*p.heads+h,0) s) μ)
      (Real.bAt p (start (grid p) (n/p.bm,b*p.heads+h,0) s) μ)
      ((start (grid p) (n/p.bm,b*p.heads+h,0) s).pids 0*p.bm+(lane p hbm n d hd).1.val)
      (lane p hbm n d hd).2.1.val=target p μ b h n d := by
  have hp : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hn : n/p.bm*p.bm+n%p.bm=n := (by simpa only [Nat.mul_comm] using Nat.div_add_mod n p.bm)
  unfold target Real.qAt Real.kAt Real.vAt Real.bAt
  simp [base,start,ids,lane,offset,hdiv,hmod,hn,Nat.mod_eq_of_lt hh]
  congr 1 <;> funext j e <;> congr 1 <;> omega

theorem returned_real (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : Physical.Store Real.R) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity) (hbm : 0<p.bm) (μ : RegionName→ Nat→ℝ)
    (reads : Real.Reads p before.extend μ) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p abi before work=some result ∧
      result.result=() ∧ result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile
            (ofFloat p.storage.float (some (target p μ b h n d)))) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=before.cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases := by
  obtain ⟨result,hr,hn,hcap,hv,hf,ha,hc⟩:=Physical.finite_return
    (fun _=>Real.algebra) (fun _=>Real.dotModel) p bases domain abi before storage fits hbm work legal
  refine ⟨result,hr,hn,hcap,?_,hf,ha,hc⟩
  intro b hb h hh n hn d hd
  obtain ⟨index,hi,hv⟩:=hv b hb h hh n hn d hd
  refine ⟨index,hi,?_⟩
  have hreads : Real.Reads p (start (grid p) (n/p.bm,b*p.heads+h,0) before.extend) μ := reads
  have active : Source.active p (start (grid p) (n/p.bm,b*p.heads+h,0) before.extend) (lane p hbm n d hd) := by
    change n/p.bm*p.bm+n%p.bm<p.length
    simpa only [Nat.mul_comm, Nat.div_add_mod] using hn
  have live : Live p := ⟨by omega,by omega,by omega⟩
  rw [Real.answer_real p _ μ hreads _ (domain.launch live).bn active,
    target_at_program p before.extend μ hbm b h n d hh hd] at hv
  exact hv

#axiomsClean input_values_reads
#axiomsClean returned_real
#specNonCircular target avoiding [FP.Structural.exec,Whole.invoke,Physical.invoke,Whole.answer]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.RealReturned
