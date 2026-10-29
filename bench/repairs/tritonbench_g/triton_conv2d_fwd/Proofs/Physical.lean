import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Native

namespace VeriTile.Bench.Repairs.Conv2d.Physical
open Native
abbrev Heap := Nat → Byte

structure Layout where
  inputBase : Nat
  weightBase : Nat
  outputBase : Nat
  inputBytes : Nat
  weightBytes : Nat
  outputWords : Nat

def Separated (a n b m : Nat) : Prop := n=0 ∨ m=0 ∨ a+n≤b ∨ b+m≤a

structure Fresh (l : Layout) : Prop where
  input : Separated l.inputBase l.inputBytes l.outputBase (4*l.outputWords)
  weight : Separated l.weightBase l.weightBytes l.outputBase (4*l.outputWords)

/-- Bounded byte observations at actual pointer bases, with inert unused
region addresses. Input and weight may share any physical bytes. -/
def observe (l : Layout) (heap : Heap) : Bytes
  | .input,a => if a<l.inputBytes then heap (l.inputBase+a) else 0
  | .weight,a => if a<l.weightBytes then heap (l.weightBase+a) else 0
  | .output,a => if a<4*l.outputWords then heap (l.outputBase+a) else 0
  | .other _,_ => 0

/-- Four concrete byte writes. The test uses complete physical addresses. -/
def write (heap : Heap) (address : Nat) (value : Word) : Heap := fun q =>
  if inside : address≤q ∧ q<address+4 then value ⟨q-address,by omega⟩ else heap q

theorem write_byte (heap : Heap) (a : Nat) (value : Word) (k : Fin 4) :
    write heap a value (a+k.val)=value k := by
  have hk:=k.isLt
  simp [write,show a≤a+k.val ∧ a+k.val<a+4 by omega]

theorem write_frame (heap : Heap) (a : Nat) (value : Word) (q : Nat)
    (outside : q<a ∨ a+4≤q) : write heap a value q=heap q := by
  simp [write,show ¬(a≤q ∧ q<a+4) by omega]

theorem separated_byte (a n b m q r : Nat) (h : Separated a n b m)
    (hq : q<n) (hr : r<m) : a+q≠b+r := by unfold Separated at h;omega

/-- Observation commutes with each actual store, using proven span
separation, independently of numerical operations and input aliasing. -/
theorem observe_write (l : Layout) (fresh : Fresh l) (heap : Heap)
    (a : Nat) (ha : a<l.outputWords) (value : Word) :
    observe l (write heap (l.outputBase+4*a) value)=byteStore (observe l heap) a value := by
  funext region q
  cases region with
  | input =>
    simp only [observe,byteStore,reduceCtorEq,false_and,↓reduceIte]
    split_ifs with inside
    · rw [write_frame]
      have h:=fresh.input
      unfold Separated at h
      omega
    · rfl
  | weight =>
    simp only [observe,byteStore,reduceCtorEq,false_and,↓reduceIte]
    split_ifs with inside
    · rw [write_frame]
      have h:=fresh.weight
      unfold Separated at h
      omega
    · rfl
  | other id => simp [observe,byteStore]
  | output =>
    simp only [observe,byteStore,true_and]
    by_cases hit:q/4=a
    · have bound:q<4*l.outputWords := by omega
      have reconstruct:l.outputBase+q=l.outputBase+4*a+q%4 := by omega
      simp only [hit,ite_true,bound]
      rw [reconstruct]
      exact write_byte heap _ value ⟨q%4,Nat.mod_lt _ (by decide)⟩
    · simp only [hit,ite_false]
      by_cases bound:q<4*l.outputWords
      · simp only [bound,ite_true]
        apply write_frame
        omega
      · simp [bound]

def commit (l : Layout) (writes : List (Nat×Word)) (heap : Heap) : Heap :=
  writes.foldl (fun h w=>write h (l.outputBase+4*w.1) w.2) heap

theorem observe_commit (l : Layout) (fresh : Fresh l) (writes : List (Nat×Word))
    (bound : ∀ w∈writes,w.1<l.outputWords) (heap : Heap) :
    observe l (commit l writes heap)=byteCommit writes (observe l heap) := by
  induction writes generalizing heap with
  | nil=>rfl
  | cons w ws ih=>
    change observe l (commit l ws (write heap (l.outputBase+4*w.1) w.2))=
      byteCommit ws (byteStore (observe l heap) w.1 w.2)
    rw [ih (fun x hx=>bound x (by simp [hx])),observe_write l fresh heap w.1 (bound w (by simp))]

theorem programWrites_bound (s : Shape) (v : Valid s) (p : Instructions)
    (flags : Flags) (pid : Program) (m : Bytes) :
    ∀ w∈programWrites s (isa p) flags pid (load m),w.1<s.size := by
  intro w hw
  obtain ⟨lane,hl,x,eq,hx,rfl⟩:=(programWrites_member s v (isa p) flags pid (load m) w).mp hw
  exact address_bound s v x hx

def program (s : Shape) (l : Layout) (p : Instructions) (flags : Flags)
    (pid : Program) (heap : Heap) :=
  commit l (programWrites s (isa p) flags pid (load (observe l heap))) heap

/-- Each program loads the evolving physical heap, then emits exactly the
source program's masked stores; no result tensor or snapshot is substituted. -/
def run (s : Shape) (l : Layout) (p : Instructions) (flags : Flags) : List Program → Heap → Heap
  | [],heap=>heap
  | pid::rest,heap=>run s l p flags rest (program s l p flags pid heap)
def execute (s : Shape) (l : Layout) (p : Instructions) (flags : Flags) (heap : Heap) :=
  run s l p flags (programs s) heap

theorem observe_program (s : Shape) (v : Valid s) (l : Layout) (fresh : Fresh l)
    (size : l.outputWords=s.size) (p : Instructions) (flags : Flags) (pid : Program) (heap : Heap) :
    observe l (program s l p flags pid heap)=byteProgram s p flags pid (observe l heap) := by
  apply observe_commit l fresh
  simpa only [size] using programWrites_bound s v p flags pid (observe l heap)

theorem observe_run (s : Shape) (v : Valid s) (l : Layout) (fresh : Fresh l)
    (size : l.outputWords=s.size) (p : Instructions) (flags : Flags) (ps : List Program) (heap : Heap) :
    observe l (run s l p flags ps heap)=bytePrograms s p flags ps (observe l heap) := by
  induction ps generalizing heap with
  | nil=>rfl
  | cons pid ps ih=>
    change observe l (run s l p flags ps (program s l p flags pid heap))=_
    rw [ih,observe_program s v l fresh size]
    rfl

theorem execute_read (s : Shape) (v : Valid s) (l : Layout) (fresh : Fresh l)
    (size : l.outputWords=s.size) (p : Instructions) (flags : Flags) (heap : Heap)
    (x : Cell) (hx : s.Active x) (k : Fin 4) :
    execute s l p flags heap (l.outputBase+4*outputPointer s x+k.val)=
      ordered s (isa p) flags (load (observe l heap)) x k := by
  have h:=selected_read s v p flags (observe l heap) x hx k
  rw [selected,←observe_run s v l fresh size] at h
  have hb:4*outputPointer s x+k.val<4*l.outputWords := by
    have ha:=address_bound s v x hx
    have hk:=k.isLt
    omega
  simpa only [observe,hb,ite_true,Nat.add_assoc,execute] using h

theorem commit_frame (l : Layout) (writes : List (Nat×Word))
    (bound : ∀ w∈writes,w.1<l.outputWords) (heap : Heap) (q : Nat)
    (outside : q<l.outputBase ∨ l.outputBase+4*l.outputWords≤q) :
    commit l writes heap q=heap q := by
  induction writes generalizing heap with
  | nil=>rfl
  | cons w ws ih=>
    change commit l ws (write heap (l.outputBase+4*w.1) w.2) q=heap q
    rw [ih (fun x hx=>bound x (by simp [hx]))]
    apply write_frame
    have hw:=bound w (by simp)
    omega

theorem run_frame (s : Shape) (v : Valid s) (l : Layout) (size : l.outputWords=s.size)
    (p : Instructions) (flags : Flags) (ps : List Program) (heap : Heap) (q : Nat)
    (outside : q<l.outputBase ∨ l.outputBase+4*l.outputWords≤q) :
    run s l p flags ps heap q=heap q := by
  induction ps generalizing heap with
  | nil=>rfl
  | cons pid ps ih=>
    change run s l p flags ps (program s l p flags pid heap) q=heap q
    rw [ih]
    apply commit_frame _ _ _ _ _ outside
    simpa only [size] using programWrites_bound s v p flags pid (observe l heap)

/-- Complete caller frame, including storage gaps, input/weight allocations,
and all output-exterior bytes. -/
theorem execute_frame (s : Shape) (v : Valid s) (l : Layout) (size : l.outputWords=s.size)
    (p : Instructions) (flags : Flags) (heap : Heap) (q : Nat)
    (outside : q<l.outputBase ∨ l.outputBase+4*l.outputWords≤q) :
    execute s l p flags heap q=heap q := run_frame s v l size p flags _ heap q outside

#axiomsClean observe_write
#axiomsClean observe_run
#axiomsClean execute_read
#axiomsClean execute_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Conv2d.Physical
