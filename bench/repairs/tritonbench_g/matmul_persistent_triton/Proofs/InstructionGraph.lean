import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Execution

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Floating
noncomputable section
set_option maxHeartbeats 5000000
abbrev Word := BitVec 32

/-- Concrete fixed ISA instruction shapes. None has the 128-row source-dot
shape, and every instruction has K≤32, below either source BK64/BK128. -/
inductive ISA where
  | wgmmaF16 | wgmmaE4M3 | mmaF16 | mmaE4M3
  deriving DecidableEq

def ISA.m : ISA → Nat | .wgmmaF16 | .wgmmaE4M3 => 64 | .mmaF16 | .mmaE4M3 => 16
def ISA.n : ISA → Nat | .wgmmaF16 | .wgmmaE4M3 => 256 | .mmaF16 | .mmaE4M3 => 8
def ISA.k : ISA → Nat | .wgmmaF16 | .mmaF16 => 16 | .wgmmaE4M3 | .mmaE4M3 => 32
def ISA.storage : ISA → Storage | .wgmmaF16 | .mmaF16 => .fp16 | .wgmmaE4M3 | .mmaE4M3 => .f8e4
def ISA.name : ISA → String
  | .wgmmaF16 => "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16"
  | .wgmmaE4M3 => "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3"
  | .mmaF16 => "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  | .mmaE4M3 => "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"

abbrev ISA.Left (op : ISA) (α : Type) := Fin op.m → Fin op.k → α
abbrev ISA.Right (op : ISA) (α : Type) := Fin op.k → Fin op.n → α
abbrev ISA.Acc (op : ISA) (α : Type) := Fin op.m → Fin op.n → α

/-- Scalar expression markers retain the complete supplied-accumulator dot.
There is deliberately no numerical evaluator for this whole-dot syntax. -/
inductive Expr where
  | raw (word : Word)
  | zero
  | cast (profile : Storage) (value : Expr)
  | dot (width : Nat) (a : Fin 128 → Fin width → Expr)
      (b : Fin width → Fin 256 → Expr) (acc : Source.Acc Expr) (row : Fin 128) (column : Fin 256)

def arithmetic : Source.Arithmetic Expr Expr where
  inputZero := .zero
  accumulatorZero := .zero
  dot := Expr.dot
  cast := Expr.cast

/-- Actual numerical graph has only output conversions and fixed ISA MMA
instructions at their ordered matrix/accumulator operands. -/
inductive Graph where
  | raw (word : Word)
  | cast (profile : Storage) (value : Graph)
  | instruction (op : ISA) (a : op.Left Graph) (b : op.Right Graph) (acc : op.Acc Graph)
      (row : Fin op.m) (column : Fin op.n)

structure Numerics where
  cast : Storage → Word → Word
  instruction : (op : ISA) → op.Left Word → op.Right Word → op.Acc Word → op.Acc Word

def Graph.eval (numerics : Numerics) : Graph → Word
  | .raw word => word
  | .cast profile value => numerics.cast profile (value.eval numerics)
  | .instruction op a b acc row column => numerics.instruction op
      (fun i k => (a i k).eval numerics) (fun k j => (b k j).eval numerics)
      (fun i j => (acc i j).eval numerics) row column

structure Domain where
  cast : Storage → Word → Prop
  instruction : (op : ISA) → op.Left Word → op.Right Word → op.Acc Word → Prop

/-- The only deferred numerical equalities: one actual output conversion,
or one named, fixed-size ISA instruction. No tl.dot or whole-output port. -/
structure LocalPorts (native reference : Numerics) (domain : Domain) : Prop where
  cast : ∀ profile word,domain.cast profile word → native.cast profile word=reference.cast profile word
  instruction : ∀ op a b acc,domain.instruction op a b acc →
    native.instruction op a b acc=reference.instruction op a b acc

def Graph.Admitted (native : Numerics) (domain : Domain) : Graph → Prop
  | .raw _ => True
  | .cast profile value => value.Admitted native domain ∧ domain.cast profile (value.eval native)
  | .instruction op a b acc _ _ =>
      (∀ i k,(a i k).Admitted native domain) ∧ (∀ k j,(b k j).Admitted native domain) ∧
      (∀ i j,(acc i j).Admitted native domain) ∧
      domain.instruction op (fun i k => (a i k).eval native) (fun k j => (b k j).eval native)
        (fun i j => (acc i j).eval native)

theorem Graph.equivalent (native reference : Numerics) (domain : Domain)
    (ports : LocalPorts native reference domain) (graph : Graph) (admitted : graph.Admitted native domain) :
    graph.eval native=graph.eval reference := by
  induction graph with
  | raw => rfl
  | cast profile value ih =>
    change native.cast profile (value.eval native)=reference.cast profile (value.eval reference)
    rw [ports.cast profile _ admitted.2,ih admitted.1]
  | instruction op a b acc row column ia ib ic =>
    change native.instruction op _ _ _ row column=reference.instruction op _ _ _ row column
    rw [ports.instruction op _ _ _ admitted.2.2.2]
    have aa : (fun i k => (a i k).eval native)=(fun i k => (a i k).eval reference) := by
      funext i k; exact ia i k (admitted.1 i k)
    have bb : (fun k j => (b k j).eval native)=(fun k j => (b k j).eval reference) := by
      funext k j; exact ib k j (admitted.2.1 k j)
    have cc : (fun i j => (acc i j).eval native)=(fun i j => (acc i j).eval reference) := by
      funext i j; exact ic i j (admitted.2.2.1 i j)
    rw [aa,bb,cc]

theorem fixed_instruction_limits (op : ISA) : op.m<128 ∧ op.k≤32 ∧ op.k<op.storage.blockK := by
  cases op <;> decide

#axiomsClean Graph.equivalent
#specNonCircular LocalPorts avoiding [Source.run,Source.accumulator,Expr.dot]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Floating
