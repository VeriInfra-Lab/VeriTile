import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Source
namespace VeriTile.Bench.Repairs.MatmulKernel.Native
abbrev Word := BitVec 32
abbrev Half := BitVec 16
structure Instructions where
  mma : (o : Opcode) → Fin o.rows → Fin o.cols → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
  narrow : Word → Half
  widen : Half → Word
/-- Only one fixed K=8 instruction result lane, and two scalar conversions.
No tensor dot, K-reduction, execution or output equality is deferred. -/
structure Obligations (actual target : Instructions) : Prop where
  mma : ∀ o i j a b acc, actual.mma o i j a b acc=target.mma o i j a b acc
  narrow : ∀ x, actual.narrow x=target.narrow x
  widen : ∀ x, actual.widen x=target.widen x
inductive Expr where
  | bits (word : Word)
  | mma (o : Opcode) (i : Fin o.rows) (j : Fin o.cols) (a b : Fin 8 → Expr) (acc : Expr)
  | cast (acc : Expr)
def Expr.eval (p : Instructions) : Expr → Word
  | .bits x => x
  | .mma o i j a b acc => p.mma o i j (fun e => (a e).eval p) (fun e => (b e).eval p) (acc.eval p)
  | .cast x => p.widen (p.narrow (x.eval p))
def graphOps : Ops Expr := ⟨.bits 0,Expr.mma,Expr.cast⟩
def wordOps (p : Instructions) : Ops Word := ⟨0,p.mma,fun x => p.widen (p.narrow x)⟩
def graphMemory (m : Memory Word) : Memory Expr := fun r a => .bits (m r a)
def graph (c : Config) (m : Memory Word) (w : Work) : Expr := value c graphOps (graphMemory m) w

theorem atom_eval (c : Config) (p : Instructions) (m : Memory Word) (w : Work) (q j : Nat) (z : Expr) :
    (atom c graphOps (graphMemory m) w q j z).eval p=atom c (wordOps p) m w q j (z.eval p) := by
  simp only [atom,graphOps,Expr.eval,wordOps]
  congr 1 <;> funext e <;> simp only [readA,readB] <;> split <;> rfl
theorem atoms_eval (c : Config) (p : Instructions) (m : Memory Word) (w : Work) (q : Nat)
    (js : List Nat) (z : Expr) :
    (js.foldl (fun z j => atom c graphOps (graphMemory m) w q j z) z).eval p=
      js.foldl (fun z j => atom c (wordOps p) m w q j z) (z.eval p) := by
  induction js generalizing z with
  | nil => rfl
  | cons j js ih => simpa only [List.foldl_cons,atom_eval] using ih (atom c graphOps (graphMemory m) w q j z)
theorem block_eval (c : Config) (p : Instructions) (m : Memory Word) (w : Work) (q : Nat) (z : Expr) :
    (block c graphOps (graphMemory m) w q z).eval p=block c (wordOps p) m w q (z.eval p) :=
  atoms_eval c p m w q _ z
theorem blocks_eval (c : Config) (p : Instructions) (m : Memory Word) (w : Work)
    (qs : List Nat) (z : Expr) :
    (qs.foldl (fun z q => block c graphOps (graphMemory m) w q z) z).eval p=
      qs.foldl (fun z q => block c (wordOps p) m w q z) (z.eval p) := by
  induction qs generalizing z with
  | nil => rfl
  | cons q qs ih => simpa only [List.foldl_cons,block_eval] using ih (block c graphOps (graphMemory m) w q z)
theorem graph_value (c : Config) (p : Instructions) (m : Memory Word) (w : Work) :
    (graph c m w).eval p=value c (wordOps p) m w := by
  change p.widen (p.narrow (((List.range c.steps).foldl
    (fun z q => block c graphOps (graphMemory m) w q z) (.bits 0)).eval p))=_
  rw [blocks_eval]
  rfl

theorem local_transport (actual target : Instructions) (h : Obligations actual target) (e : Expr) :
    e.eval actual=e.eval target := by
  induction e with
  | bits w => rfl
  | mma o i j a b acc ia ib ic => simp only [Expr.eval,h.mma,ia,ib,ic]
  | cast acc ih => simp only [Expr.eval,h.widen,h.narrow,ih]
theorem value_transport (c : Config) (actual target : Instructions) (h : Obligations actual target)
    (m : Memory Word) (w : Work) : value c (wordOps actual) m w=value c (wordOps target) m w := by
  rw [←graph_value,←graph_value]
  exact local_transport actual target h _

/-- Independent native contract uses caller coordinates and explicitly
ordered fixed-K instruction calls. Both casts are retained. -/
def reference (c : Config) (p : Instructions) (m : Memory Word) (r n : Nat) : Word :=
  p.widen (p.narrow ((List.range (ceil c.K c.BK)).foldl (fun acc q =>
    (List.range (c.BK/8)).foldl (fun acc j =>
      p.mma c.opcode ⟨(r%c.BM)%c.opcode.rows,Nat.mod_lt _ c.opcode.rows_pos⟩
        ⟨(n%c.BN)%c.opcode.cols,Nat.mod_lt _ c.opcode.cols_pos⟩
        (fun e => if q*c.BK+j*8+e.val<c.K then m .a (c.A.addr r (q*c.BK+j*8+e.val)) else 0)
        (fun e => if q*c.BK+j*8+e.val<c.K then m .b (c.B.addr (q*c.BK+j*8+e.val) n) else 0) acc) acc) 0))
theorem value_reference (c : Config) (p : Instructions) (m : Memory Word) (w : Work) (hw : c.Active w) :
    value c (wordOps p) m w=reference c p m w.1 w.2 := by
  simp only [value,block,atom,wordOps,readA,readB,Nat.mod_eq_of_lt hw.1,Nat.mod_eq_of_lt hw.2,Config.steps,reference]
theorem native_source_readback (c : Config) (hc : c.Valid) (p : Instructions) (m : Memory Word)
    (w : Work) (hw : c.Active w) :
    source c (wordOps p) m .c (c.C.addr w.1 w.2)=reference c p m w.1 w.2 := by
  rw [source_run,run_readback c hc _ m w hw,value_reference c p m w hw]
#specNonCircular reference avoiding [graph,value,atom,block,source,run]
#axiomsClean graph_value
#axiomsClean local_transport
#axiomsClean native_source_readback
end VeriTile.Bench.Repairs.MatmulKernel.Native
