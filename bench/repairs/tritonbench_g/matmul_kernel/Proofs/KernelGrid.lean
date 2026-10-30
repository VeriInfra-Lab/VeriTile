import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelComplete
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Scheduling
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def atProgram (s : State α) (pid : Program) : State α :=
  {s with pids:=fun d=>if d=0 then pid.1 else if d=1 then pid.2 else s.pids d}
def owner (p : Config) (w : Work) : Program := (w.1/p.BM,w.2/p.BN)
def lane (p : Config) (hc : p.Valid) (w : Work) : CIndex p :=
  (⟨w.1%p.BM,Nat.mod_lt _ hc.bm⟩,⟨w.2%p.BN,Nat.mod_lt _ hc.bn⟩,PUnit.unit)
def point (M : Algebra α) (D : DotAcc α) (p : Config) (hc : p.Valid) (s : State α) (w : Work) : α :=
  outputValue M D p (atProgram s (owner p w)) (lane p hc w)

theorem own_row (p : Config) (hc : p.Valid) (s : State α) (w : Work) :
    row p (atProgram s (owner p w)) (lane p hc w).1=w.1 := by
  simp only [row,pidM,atProgram,owner,lane,BM,ite_true]
  simpa only [Nat.mul_comm] using Nat.div_add_mod w.1 p.BM
theorem own_col (p : Config) (hc : p.Valid) (s : State α) (w : Work) :
    col p (atProgram s (owner p w)) (lane p hc w).2.1=w.2 := by
  simp only [col,pidN,atProgram,owner,lane,BN,show ¬(1:Nat)=0 by decide,ite_false,ite_true]
  simpa only [Nat.mul_comm] using Nat.div_add_mod w.2 p.BN

theorem tile_injective (p : Config) (hc : p.Valid) (s : State α)
    (i j : CIndex p) (hi : active p s i) (hj : active p s j)
    (eq : cOffset p s i=cOffset p s j) : i=j := by
  have same:=c_injective p hc (row p s i.1,col p s i.2.1)
    (row p s j.1,col p s j.2.1) hi hj (by simpa [cOffset,View.addr,Nat.mul_comm] using eq)
  have rm:=congrArg Prod.fst same
  have cn:=congrArg Prod.snd same
  have ri : i.1=j.1 := Fin.ext (by simp only [row] at rm;omega)
  have ci : i.2.1=j.2.1 := Fin.ext (by simp only [col] at cn;omega)
  exact Prod.ext ri (Prod.ext ci (Subsingleton.elim _ _))

theorem at_correct (M : Algebra α) (D : DotAcc α) (p : Config) (hc : p.Valid)
    (s : State α) (pid : Program) :
    ∃ t,FP.Structural.exec (backend M D) (source p) (atProgram s pid)=some t ∧
      (∀ w,p.Active w → owner p w=pid →
        t.mem "C" (p.C.addr w.1 w.2)=stored (point M D p hc s w)) ∧
      (∀ (r : RegionName) a,(r≠"C" ∨ ∀ w,p.Active w → owner p w=pid → a≠p.C.addr w.1 w.2) →
        t.mem r a=s.mem r a) := by
  obtain ⟨t,he,_,readback,frame⟩:=correctness M D p (atProgram s pid) (tile_injective p hc _)
  refine ⟨t,he,?_,?_⟩
  · intro w hw own
    have active : active p (atProgram s pid) (lane p hc w) := by
      rw [←own]
      simpa only [active,own_row,own_col] using hw
    have off : cOffset p (atProgram s pid) (lane p hc w)=p.C.addr w.1 w.2 := by
      rw [←own]
      simp only [cOffset,own_row,own_col,View.addr,Nat.mul_comm]
    rw [←off,readback _ active]
    simp only [point,own]
  · intro r a outside
    apply frame
    rcases outside with outside|outside
    · exact Or.inl outside
    · right
      intro i hi
      let w : Work := (row p (atProgram s pid) i.1,col p (atProgram s pid) i.2.1)
      have own : owner p w=pid := by
        apply Prod.ext
        · exact tile_div pid.1 i.1.val p.BM hc.bm i.1.isLt
        · exact tile_div pid.2 i.2.1.val p.BN hc.bn i.2.1.isLt
      simpa [w,cOffset,View.addr,Nat.mul_comm] using outside w hi own

theorem point_stable (M : Algebra α) (D : DotAcc α) (p : Config) (hc : p.Valid)
    (s t : State α) (w : Work)
    (ha : ∀ o,t.mem "A" o=s.mem "A" o) (hb : ∀ o,t.mem "B" o=s.mem "B" o) :
    point M D p hc t w=point M D p hc s w := by
  have same : update M D p (atProgram t (owner p w))=update M D p (atProgram s (owner p w)) := by
    funext q acc
    unfold update
    congr 1
    · funext i
      simp [aValue,aOffset,aRow,row,pidM,atProgram,ha]
    · funext i
      simp [bValue,bOffset,bCol,col,pidN,atProgram,hb]
  simp only [point,outputValue,accumulation,same]

def execute (M : Algebra α) (D : DotAcc α) (p : Config) : List Program → State α → Option (State α)
  | [],s=>some s
  | pid::tail,s=>(FP.Structural.exec (backend M D) (source p) (atProgram s pid)).bind (execute M D p tail)

theorem order_correct (M : Algebra α) (D : DotAcc α) (p : Config) (hc : p.Valid)
    (order : List Program) (s : State α) :
    ∃ t,execute M D p order s=some t ∧
      (∀ w,p.Active w → owner p w∈order →
        t.mem "C" (p.C.addr w.1 w.2)=stored (point M D p hc s w)) ∧
      (∀ (r : RegionName) a,(r≠"C" ∨ ∀ w,p.Active w → owner p w∈order → a≠p.C.addr w.1 w.2) →
        t.mem r a=s.mem r a) := by
  induction order generalizing s with
  | nil=>
    refine ⟨s,rfl,?_,?_⟩
    · intro w hw hm;simp at hm
    · intros;rfl
  | cons pid tail ih=>
    obtain ⟨u,hu,uread,uframe⟩:=at_correct M D p hc s pid
    obtain ⟨t,ht,tread,tframe⟩:=ih u
    have same (w : Work) : point M D p hc u w=point M D p hc s w :=
      point_stable M D p hc s u w
        (fun o=>uframe "A" o (Or.inl (by decide)))
        (fun o=>uframe "B" o (Or.inl (by decide)))
    refine ⟨t,by simp only [execute,hu,Option.bind_some,ht],?_,?_⟩
    · intro w hw hm
      by_cases member : owner p w∈tail
      · rw [tread w hw member,same]
      · have own : owner p w=pid := by simpa only [List.mem_cons,member,or_false] using hm
        rw [tframe]
        · exact uread w hw own
        · right
          intro z hz memberZ equal
          have wz:=c_injective p hc w z hw hz equal
          exact member (by simpa only [←wz] using memberZ)
    · intro r a outside
      rw [tframe]
      · apply uframe
        exact outside.imp id (fun h w hw own=>h w hw (by simp [own]))
      · exact outside.imp id (fun h w hw own=>h w hw (by simp [own]))

def launch (M : Algebra α) (D : DotAcc α) (p : Config) (s : State α) :=
  execute M D p (programs p) s

theorem launch_correct (M : Algebra α) (D : DotAcc α) (p : Config) (hc : p.Valid) (s : State α) :
    ∃ t,launch M D p s=some t ∧
      (∀ w,p.Active w → t.mem "C" (p.C.addr w.1 w.2)=stored (point M D p hc s w)) ∧
      (∀ (r : RegionName) a,(r≠"C" ∨ ∀ w,p.Active w → a≠p.C.addr w.1 w.2) → t.mem r a=s.mem r a) := by
  obtain ⟨t,he,read,frame⟩:=order_correct M D p hc (programs p) s
  refine ⟨t,he,?_,?_⟩
  · intro w hw
    apply read w hw
    have hm:=coordinate_owner w.1 p.BM p.M hc.bm hw.1
    have hn:=coordinate_owner w.2 p.BN p.N hc.bn hw.2
    simpa only [programs,owner,List.mem_product,List.mem_range] using And.intro hm.1 hn.1
  · intro r a outside
    apply frame
    exact outside.imp id (fun h w hw _=>h w hw)

theorem empty_launch (M : Algebra α) (D : DotAcc α) (p : Config) (s : State α)
    (empty : p.M=0 ∨ p.N=0) : launch M D p s=some s := by
  have zeroceil (b : Nat) : ceil 0 b=0 := by
    by_cases hz : b=0
    · simp [ceil,hz]
    · exact Nat.div_eq_of_lt (by omega)
  rcases empty with empty|empty <;> simp [launch,programs,empty,zeroceil,execute]

#axiomsClean at_correct
#axiomsClean order_correct
#axiomsClean launch_correct
#axiomsClean empty_launch
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelSource
