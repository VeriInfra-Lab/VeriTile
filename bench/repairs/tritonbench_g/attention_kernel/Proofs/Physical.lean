import bench.repairs.tritonbench_g.attention_kernel.Proofs.Access
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Physical
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000

/-- Actual finite typed word allocations; both supported formats are two bytes.
No new tensor or backing allocation is created by this caller-output kernel. -/
structure Store (α : Type) where
  capacity : RegionName → Nat
  cells : (r : RegionName) → Fin (capacity r) → Cell α
  outside : Cell α

def Store.extend (m : Store α) : State α where
  mem:=fun r off=>if h : off<m.capacity r then m.cells r ⟨off,h⟩ else m.outside
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>1
  undef:=fun d _ _=>defaultValue d

def restrict (capacity : RegionName → Nat) (s : State α) : Store α :=
  ⟨capacity,(fun r i=>s.mem r i.val),.mk .real default⟩

structure Returned (α : Type) where
  memory : Store α
  result : Unit

def invoke (M : Program→ Algebra α) (dot : Program→ Dot α) (p : Parameters)
    (abi : InputABI p) (before : Store α) (work : List Program) : Option (Returned α) :=
  (Whole.invoke M dot p abi before.extend work).map fun result=>
    ⟨restrict before.capacity result.state,result.result⟩

/-- Every location outside the logical caller output, including view gaps. -/
def Exterior (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  r≠p.Out ∨ ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.dim,
    off≠logicalAddress p b h n d

theorem exterior_outside (p : Parameters) (s : State α) (work : List Program)
    (legal : (programs (grid p)).Perm work) (r : RegionName) (off : Nat) (outside : Exterior p r off) :
    Outside p (grid p) work s r off := by
  rcases outside with hr|hout
  · exact Or.inl hr
  · right
    intro pid hp i hi
    rcases pid with ⟨x,y,z⟩
    have hmem:=(mem_programs (grid p) x y z).mp (legal.mem_iff.mpr hp)
    obtain ⟨_,hb,hh⟩:=head_decode p y hmem.2.1
    exact hout _ hb _ hh _ hi _ i.2.1.isLt

theorem finite_return (M : Program→ Algebra α) (dot : Program→ Dot α) (p : Parameters)
    (bases : Bases) (domain : Domain p bases) (abi : InputABI p)
    (before : Store α) (storage : TensorStorage p) (fits : storage.capacity=before.capacity)
    (hbm : 0<p.bm) (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke M dot p abi before work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile
            (ofFloat p.storage.float (answer (M (n/p.bm,b*p.heads+h,0)) (dot (n/p.bm,b*p.heads+h,0)) p
              (start (grid p) (n/p.bm,b*p.heads+h,0) before.extend) (lane p hbm n d hd)))) ∧
      (∀ r (i : Fin (before.capacity r)),Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=before.cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases := by
  by_cases live : Live p
  · obtain ⟨ret,hr,hn,hv,hf⟩:=host_result M dot p live (domain.launch live) storage abi before.extend work legal
    let result : Returned α:=⟨restrict before.capacity ret.state,ret.result⟩
    refine ⟨result,by simp [invoke,hr,result],hn,rfl,?_,?_,
      (fun l=>Access.safe p (domain.launch l) storage),Machine.closed p bases domain⟩
    · intro b hb h hh n hn d hd
      have bound:=storage.bounds p.Out p.o (by simp) b hb h hh n hn d hd
      rw [fits] at bound
      refine ⟨⟨logicalAddress p b h n d,bound⟩,rfl,?_⟩
      exact hv b hb h hh n hn d hd
    · intro r i hi
      refine ⟨i,rfl,?_⟩
      change ret.state.mem r i.val=before.cells r i
      rw [hf r i.val (exterior_outside p before.extend work legal r i.val hi)]
      simp [Store.extend,i.isLt]
  · let result : Returned α:=⟨restrict before.capacity before.extend,()⟩
    refine ⟨result,by simp [invoke,Whole.invoke,live,result],rfl,rfl,?_,?_,
      (fun l=>False.elim (live l)),Machine.closed p bases domain⟩
    · intro b hb h hh n hn d hd
      exact False.elim (live ⟨by omega,by omega,by omega⟩)
    · intro r i hi
      refine ⟨i,rfl,?_⟩
      simp [result,restrict,Store.extend,i.isLt]

#axiomsClean finite_return
#specNonCircular Exterior avoiding [FP.Structural.exec,Whole.invoke]
#specNonCircular Store avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Physical
