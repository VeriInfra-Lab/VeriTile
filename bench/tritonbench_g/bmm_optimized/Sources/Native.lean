import bench.tritonbench_g.bmm_optimized.Sources.Scatter

namespace VeriTile.Bench.Repairs.BmmOptimized.Native
abbrev Byte := BitVec 8
/-- Four little-endian bytes, retaining every raw FP32 representation. -/
abbrev Word := Fin 4 → Byte
abbrev Bytes := Memory Byte
def zero : Word := fun _ => 0

/-- Only scalar PTX fma.rn.f32 and host metadata-resolution negation are ports.
No dot, reduction, block, tensor or output-valued port exists. -/
structure Instructions where
  fmaRN : Word → Word → Word → Word
  neg : Word → Word
def scalar (p : Instructions) : Scalar Word := ⟨zero,p.fmaRN⟩
def load (m : Bytes) : Memory Word := fun r a i => m r (4*a+i.val)
def byteStore (m : Bytes) (address : Nat) (value : Word) : Bytes :=
  fun r a => if r=.output ∧ a/4=address then value ⟨a%4,Nat.mod_lt _ (by decide)⟩ else m r a

theorem load_store (m : Bytes) (a : Nat) (value : Word) :
    load (byteStore m a value)=store (load m) a value := by
  funext r b i
  have hd : (4*b+i.val)/4=b := by omega
  have hm : (4*b+i.val)%4=i.val := by omega
  simp only [load,byteStore,store,hd]
  split_ifs <;> simp_all [load]
  congr 1
  exact Fin.ext (Nat.mod_eq_of_lt i.isLt)

def byteCommit (writes : List (Nat×Word)) (m : Bytes) :=
  writes.foldl (fun state w => byteStore state w.1 w.2) m
theorem load_commit (writes : List (Nat×Word)) (m : Bytes) :
    load (byteCommit writes m)=commit writes (load m) := by
  induction writes generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change load (byteCommit ws (byteStore m w.1 w.2))=
      commit ws (store (load m) w.1 w.2)
    rw [ih,load_store]

def byteProgram (s : Shape) (c : Config) (p : Instructions) (pid : Program) (m : Bytes) :=
  byteCommit (programWrites s c (scalar p) pid (load m)) m
/-- Actual current-memory execution; each CTA computes its loads after all
previous CTA stores. Initial-snapshot equivalence is proved, not assumed. -/
def bytePrograms (s : Shape) (c : Config) (p : Instructions) : List Program → Bytes → Bytes
  | [],m => m
  | pid::ps,m => bytePrograms s c p ps (byteProgram s c p pid m)
def selected (s : Shape) (c : Config) (p : Instructions) (m : Bytes) :=
  bytePrograms s c p (programs s c) m

theorem load_program (s : Shape) (c : Config) (p : Instructions) (pid : Program) (m : Bytes) :
    load (byteProgram s c p pid m)=sourceProgram s c (scalar p) pid (load m) :=
  load_commit _ _
theorem load_programs (s : Shape) (c : Config) (p : Instructions) (ps : List Program) (m : Bytes) :
    load (bytePrograms s c p ps m)=sourcePrograms s c (scalar p) ps (load m) := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    change load (bytePrograms s c p ps (byteProgram s c p pid m))=_
    rw [ih,load_program]
    rfl
theorem load_selected (s : Shape) (c : Config) (p : Instructions) (m : Bytes) :
    load (selected s c p m)=source s c (scalar p) (load m) := load_programs s c p _ m

/-- Every byte of every active result is the independent flat padded FMA
chain, including signed zero/NaN/subnormal words without algebraic identities. -/
theorem selected_read (s : Shape) (c : Config) (p : Instructions) (m : Bytes)
    (x : Cell) (hx : s.Active x) (i : Fin 4) :
    selected s c p m .output (4*s.o x.batch x.row x.column+i.val)=
      ordered s (scalar p) (load m) x i := by
  have h := congrFun (source_grid_ordered s c (scalar p) (load m) x hx) i
  rw [←load_selected] at h
  exact h

theorem selected_frame (s : Shape) (c : Config) (p : Instructions) (m : Bytes)
    (r : Region) (address : Nat) (outside : r≠.output ∨ 4*s.size≤address) :
    selected s c p m r address=m r address := by
  have h : r≠.output ∨ s.size≤address/4 := by
    rcases outside with region|bound
    · exact Or.inl region
    · exact Or.inr (by omega)
  have eq := congrFun (source_frame s c (scalar p) (load m) r (address/4) h)
    (⟨address%4,Nat.mod_lt _ (by decide)⟩ : Fin 4)
  rw [←load_selected] at eq
  have reconstruct : 4*(address/4)+address%4=address := by omega
  simpa only [load,reconstruct] using eq

structure Obligations (actual target : Instructions) : Prop where
  fmaRN : ∀ a b carry,actual.fmaRN a b carry=target.fmaRN a b carry
  neg : ∀ a,actual.neg a=target.neg a
theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext (fun a => funext (fun b => funext (h.fmaRN a b))),funext h.neg⟩
theorem selected_equiv (s : Shape) (c : Config) (actual target : Instructions)
    (h : Obligations actual target) (m : Bytes) : selected s c actual m=selected s c target m := by
  rw [instructions_eq actual target h]

#specNonCircular scalar avoiding [source,sourceValue,sourceProgram,sourcePrograms,selected]
#axiomsClean load_store
#axiomsClean load_selected
#axiomsClean selected_read
#axiomsClean selected_frame
#axiomsClean selected_equiv
end VeriTile.Bench.Repairs.BmmOptimized.Native
