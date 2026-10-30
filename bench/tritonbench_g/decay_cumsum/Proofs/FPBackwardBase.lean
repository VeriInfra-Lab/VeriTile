import bench.tritonbench_g.decay_cumsum.Proofs.FPBackwardControl
import bench.tritonbench_g.decay_cumsum.Proofs.FPStoreFrame

namespace VeriTile.Bench.Repairs.DecayCumsum.FPBackwardBase
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps FPPrefix Geometry
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def inputRegions : List RegionName := ["dqi","dqo","dki","dko","q","k","g"]
def outputRegions : List RegionName := ["dqo","dko","dg"]
def pointerNames : List (RegName × RegionName) :=
  [("p_g","g"),("p_k","k"),("p_q","q"),("p_dq_inner","dqi"),("p_dk_inner","dki"),
   ("p_dq_inter","dqo"),("p_dk_inter","dko"),("p_dg","dg")]
def time (p : Phase) (n : Nat) : Nat := p.span-1-n

def gate (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) (t : Nat) : Values α .real [h.BK] :=
  fun i => FPReference.gate M (raw h.BK "g" (off h p prog t) u i)
def qValue (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) (t : Nat) : Values α .real [h.BK] :=
  fun i => FPReference.backwardQ M (raw h.BK "dqi" (off h p prog t) u i)
    (raw h.BK "dqo" (off h p prog t) u i) (raw h.BK "g" (off h p prog t) u i)
def kValue (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) (t : Nat) : Values α .real [h.BK] :=
  fun i => FPReference.backwardK M (raw h.BK "dki" (off h p prog t) u i)
    (raw h.BK "dko" (off h p prog t) u i) (raw h.BK "g" (off h p prog (p.span-1)) u i)
    (raw h.BK "g" (off h p prog t) u i)
def carry (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) (n : Nat) : Values α .real [h.BK] :=
  fun i => FPReference.backwardCarry M p.span
    (fun t => raw h.BK "dqi" (off h p prog t) u i) (fun t => raw h.BK "dqo" (off h p prog t) u i)
    (fun t => raw h.BK "dki" (off h p prog t) u i) (fun t => raw h.BK "dko" (off h p prog t) u i)
    (fun t => raw h.BK "q" (off h p prog t) u i) (fun t => raw h.BK "k" (off h p prog t) u i)
    (fun t => raw h.BK "g" (off h p prog t) u i) n

def output (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α)
    (r : RegionName) (t : Nat) : Values α .real [h.BK] :=
  if r="dqo" then qValue M h p prog u t else if r="dko" then kValue M h p prog u t else carry M h p prog u (p.span-t)

structure Holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤p.span
  pointers : ∀ name r, (name,r)∈pointerNames → n<p.span →
    u.regs .ptr [h.BK] name=some (ptr h.BK r (off h p prog (time p n)))
  mask : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true)
  acc : u.regs .real [h.BK] "cum_grad_dg"=some (carry M h p prog initial n)
  last : n≠0 → u.regs .real [h.BK] "last_g"=some (gate M h p prog initial (p.span-1))
  reads : ∀ r∈inputRegions, ∀ t, t<p.span-n → ∀ j : Fin h.BK,
    u.mem r (off h p prog t j)=initial.mem r (off h p prog t j)
  output : ∀ r∈outputRegions, ∀ t, p.span-n≤t → t<p.span → ∀ j : Fin h.BK,
    u.mem r (off h p prog t j)=FPReference.stored M (output M h p prog initial r t (j,PUnit.unit))
  frame : FPMemory.Frame h p prog outputRegions initial u

theorem initial_holds (M : Algebra α) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (u : State α) : Holds M h p prog u 0 (FPPrefix.initial M .backward h p prog u) := by
  have mask : (fun i : TileIndex [h.BK] => decide (prog.feature*h.BK+i.1.val<h.width))=(fun _ => Bool.true) := by
    funext i
    exact decide_eq_true (column_inside h p prog i.1)
  have last : (fun j : Fin h.BK => prog.row*h.headStride+prog.feature*h.BK+j.val+(prog.chunk*p.span+p.span-1)*h.width)=off h p prog (p.span-1) := by
    funext j
    have timeEq : prog.chunk*p.span+p.span-1=prog.chunk*p.span+(p.span-1) := by omega
    simp [off,timeEq,Nat.add_mul]
    omega
  refine ⟨Nat.zero_le _,?_,?_,?_,by simp,by intros; rfl,?_,fun _ _ _ => rfl⟩
  · intro name r member _
    simp only [pointerNames,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at member
    rcases member with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩
    all_goals simp [FPPrefix.initial,time,last]
  · simp [FPPrefix.initial,mask]
  · simp [FPPrefix.initial,carry,FPReference.backwardCarry] <;> rfl
  · intro r hr t ht ht'
    omega

theorem bind_holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M h p prog initial n u) :
    Holds M h p prog initial n (u.setReg "__rev_j" .int [] (fun _ => Int.ofNat n-Int.ofNat (p.span-1))) := by
  refine ⟨inv.bound,?_,?_,?_,?_,inv.reads,inv.output,inv.frame⟩
  · intro name r member hn
    have different : name≠"__rev_j" := by
      simp only [pointerNames,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at member
      rcases member with ⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩ <;> decide
    simpa [different] using inv.pointers name r member hn
  · simp [inv.mask]
  · simp [inv.acc]
  · intro hn
    simpa using inv.last hn

#axiomsClean initial_holds
end
end VeriTile.Bench.Repairs.DecayCumsum.FPBackwardBase
