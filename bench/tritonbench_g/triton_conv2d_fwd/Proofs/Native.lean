import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Scatter
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.Conv2d.Native
abbrev Byte := BitVec 8
/-- Four little-endian bytes, retaining every raw FP32 representation. -/
abbrev Word := Fin 4 → Byte
abbrev Bytes := Memory Byte
def zero : Word := fun _ => 0

abbrev Half := Fin 2 → Byte
/-- Raw-word ports for literal PTX instructions. TF32 truncation/input behavior
belongs to m64n32k8.f32.tf32.tf32 itself; no extra conversion is inserted. -/
structure Instructions where
  fmaRN : Word → Word → Word → Word
  halfRN : Word → Half
  wgmma16 : Mat 64 16 Half → Mat 16 32 Half → Frag Word → Frag Word
  wgmma8 : Mat 64 8 Word → Mat 8 32 Word → Frag Word → Frag Word
def isa (p : Instructions) : ISA Word Half := ⟨zero,p.fmaRN,p.halfRN,p.wgmma16,p.wgmma8⟩
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

def byteProgram (s : Shape) (p : Instructions) (flags : Flags) (pid : Program) (m : Bytes) :=
  byteCommit (programWrites s (isa p) flags pid (load m)) m
/-- Actual current-memory execution; each CTA computes its loads after all
previous CTA stores. Initial-snapshot equivalence is proved, not assumed. -/
def bytePrograms (s : Shape) (p : Instructions) (flags : Flags) : List Program → Bytes → Bytes
  | [],m => m
  | pid::ps,m => bytePrograms s p flags ps (byteProgram s p flags pid m)
def selected (s : Shape) (p : Instructions) (flags : Flags) (m : Bytes) :=
  bytePrograms s p flags (programs s) m

theorem load_program (s : Shape) (p : Instructions) (flags : Flags) (pid : Program) (m : Bytes) :
    load (byteProgram s p flags pid m)=sourceProgram s (isa p) flags pid (load m) :=
  load_commit _ _
theorem load_programs (s : Shape) (p : Instructions) (flags : Flags) (ps : List Program) (m : Bytes) :
    load (bytePrograms s p flags ps m)=sourcePrograms s (isa p) flags ps (load m) := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    change load (bytePrograms s p flags ps (byteProgram s p flags pid m))=_
    rw [ih,load_program]
    rfl
theorem load_selected (s : Shape) (p : Instructions) (flags : Flags) (m : Bytes) :
    load (selected s p flags m)=source s (isa p) flags (load m) := load_programs s p flags _ m

/-- Every byte of every active result is the independent ordered instruction
circuit, including signed zero/NaN/subnormal words without algebraic identities. -/
theorem selected_read (s : Shape) (v : Valid s) (p : Instructions) (flags : Flags) (m : Bytes)
    (x : Cell) (hx : s.Active x) (i : Fin 4) :
    selected s p flags m .output (4*outputPointer s x+i.val)=
      ordered s (isa p) flags (load m) x i := by
  have eq : source s (isa p) flags (load m) .output (outputPointer s x)=ordered s (isa p) flags (load m) x := by
    rw [source_read s v (isa p) flags (load m) x hx,source_ordered s (isa p) flags (load m) x hx.1]
  have h := congrFun eq i
  rw [←load_selected] at h
  exact h

theorem selected_frame (s : Shape) (v : Valid s) (p : Instructions) (flags : Flags) (m : Bytes)
    (r : Region) (address : Nat) (outside : r≠.output ∨ 4*s.size≤address) :
    selected s p flags m r address=m r address := by
  have h : r≠.output ∨ s.size≤address/4 := by
    rcases outside with region|bound
    · exact Or.inl region
    · exact Or.inr (by omega)
  have eq := congrFun (source_frame s v (isa p) flags (load m) r (address/4) h)
    (⟨address%4,Nat.mod_lt _ (by decide)⟩ : Fin 4)
  rw [←load_selected] at eq
  have reconstruct : 4*(address/4)+address%4=address := by omega
  simpa only [load,reconstruct] using eq

structure Obligations (actual target : Instructions) : Prop where
  fmaRN : ∀ a b carry,actual.fmaRN a b carry=target.fmaRN a b carry
  halfRN : ∀ a,actual.halfRN a=target.halfRN a
  wgmma16 : ∀ a b carry,actual.wgmma16 a b carry=target.wgmma16 a b carry
  wgmma8 : ∀ a b carry,actual.wgmma8 a b carry=target.wgmma8 a b carry
theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext (fun a => funext (fun b => funext (h.fmaRN a b))),funext h.halfRN,
    funext (fun a => funext (fun b => funext (h.wgmma16 a b))),
    funext (fun a => funext (fun b => funext (h.wgmma8 a b)))⟩
theorem selected_equiv (s : Shape) (flags : Flags) (actual target : Instructions)
    (h : Obligations actual target) (m : Bytes) : selected s actual flags m=selected s target flags m := by
  rw [instructions_eq actual target h]

#specNonCircular isa avoiding [source,sourceValue,sourceProgram,sourcePrograms,selected]
#axiomsClean load_store
#axiomsClean load_selected
#axiomsClean selected_read
#axiomsClean selected_frame
#axiomsClean selected_equiv
end VeriTile.Bench.Repairs.Conv2d.Native
