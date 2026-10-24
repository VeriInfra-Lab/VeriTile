import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Raw

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.NativeCompatibility
open VeriTile Triton FP.Structural Whole001 Source Whole Machine KernelInterface
noncomputable section
set_option maxHeartbeats 6000000

/-- State-free native instruction reference over raw input words. -/
def rawReader (words : RegionName → Nat → BitVec 32) : RegionName → Nat → Native.Expr :=
  fun r off _=>.ready (.bits (words r off))

def rawReference (I : KernelOnly.Instructions) (p : Parameters)
    (words : RegionName → Nat → BitVec 32) (b h n d : Nat) (hd : d<96) : BitVec 16 :=
  ((Native.lower (KernelOnly.Specification.algorithm Native.algebra Native.dotModel p
    (rawReader words) (n/128) b h (lane n d hd) (Native.physicalOwner d))).eval I.isa).setWidth 16

/-- Complete native grid and every BF16 result, retaining original precision,
local primitive ports, input aliases and full raw frame. -/
theorem kernel_raw_native (I : KernelOnly.Instructions) (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (before : State Native.Expr) (words : RegionName → Nat → BitVec 32)
    (reads : ∀ r∈[p.Q,p.K,p.V,p.QS,p.KS],∀ off,(before.mem r off).read .real=rawReader words r off)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel)
        p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ((Native.lower (((after.mem p.Out (offset p.o b h n d)).read .bf16)
          (Native.physicalOwner d))).eval I.isa).setWidth 16=rawReference I p words b h n d hd) ∧
      RawFrame p before after ∧ Access.Safe p ∧ Closed p bases := by
  obtain ⟨after,ran,outputs,frame⟩ := complete_raw Native.algebra Native.dotModel p bases dom sep before work order
  refine ⟨after,ran,?_,frame,Access.safe p dom.canonical,Machine.closed p bases dom⟩
  intro b hb h hh n hn d hd
  rw [outputs b hb h hh n hn d hd,Cell.read_same]
  have positive : 0<p.heads := by omega
  have div : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh]
    omega
  have mod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have algorithm := KernelOnly.source_algorithm Native.algebra Native.dotModel p
    (start (grid p) (n/128,b*p.heads+h,0) before) (rawReader words) (n/128) b h
    (by simp [start,ids]) (by simpa [start,ids] using div)
    (by simpa [start,ids] using mod) (by simp [start,ids]) reads
  rw [algorithm]
  rfl

#axiomsClean kernel_raw_native
#specNonCircular rawReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.NativeCompatibility
