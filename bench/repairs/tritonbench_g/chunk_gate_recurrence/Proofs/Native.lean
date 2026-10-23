import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Map
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section
abbrev Word := BitVec 32
structure ISA where
  add : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word

def wordOps (isa : ISA) : Ops Word := ⟨0,isa.add,isa.mul,isa.fma⟩

inductive Expr where
  | word (bits : Word)
  | zero
  | add (a b : Expr)
  | mul (a b : Expr)
  | fma (a b c : Expr)

def Expr.eval (isa : ISA) : Expr → Word
  | .word b => b
  | .zero => 0
  | .add a b => isa.add (a.eval isa) (b.eval isa)
  | .mul a b => isa.mul (a.eval isa) (b.eval isa)
  | .fma a b c => isa.fma (a.eval isa) (b.eval isa) (c.eval isa)

def graphOps : Ops Expr := ⟨.zero,.add,.mul,.fma⟩
def Expr.evalHom (isa : ISA) : OpsHom (Expr.eval isa) graphOps (wordOps isa) :=
  ⟨rfl,fun _ _ => rfl,fun _ _ => rfl,fun _ _ _ => rfl⟩

namespace Graph
def run (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) : Memory Expr :=
  Source.run graphOps c trees seed (Memory.map Expr.word m)

theorem eval_run (isa : ISA) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) :
    Memory.map (Expr.eval isa) (run c trees seed m)=Source.run (wordOps isa) c trees seed m := by
  exact Source.run_map (Expr.eval isa) graphOps (wordOps isa) (Expr.evalHom isa) c trees seed (Memory.map Expr.word m)
end Graph

namespace SourceFP
open VeriTile.Triton
def value (b : Word) : FP.Value .fp32 := ⟨b⟩
/-- Concrete RNE FP32, gradual underflow and canonical positive quiet NaN.
Native agreement is not assumed for free and is not a Real equality. -/
def scalar : ISA :=
  ⟨fun a b => (FP.add .ieeeValue (value a) (value b)).bits,
   fun a b => (FP.mul .ieeeValue (value a) (value b)).bits,
   fun a b c => (FP.fma .ieeeValue (value a) (value b) (value c)).bits⟩
/-- Local symmetry at the zero multiplicand, proved for the concrete software
profile (including NaN, infinity and signed zero). No ISA law is asserted. -/
theorem zero_fma_swap (gate incoming : Word) :
    scalar.fma gate 0 incoming = scalar.fma 0 gate incoming := by
  change (FP.fma .ieeeValue (value gate) (value 0) (value incoming)).bits =
    (FP.fma .ieeeValue (value 0) (value gate) (value incoming)).bits
  have hnan : (value 0).isNaN = Bool.false := by decide
  have hinf : (value 0).isInf = Bool.false := by decide
  have hzero : (value 0).isZero = Bool.true := by decide
  have hsign : (value 0).sign = Bool.false := by decide
  have hdecode : (value 0).decode = .finite Bool.false 0 := by
    have he : (value 0).exponent = 0 := by decide
    have hf : (value 0).fraction = 0 := by decide
    norm_num [FP.Value.decode, he, hf, hsign]
  simp only [FP.fma, FP.prepare, FP.Config.ieeeValue, Bool.false_and,
    Bool.false_eq_true, ↓reduceIte, hnan, hinf, hzero, hsign, hdecode,
    Bool.or_false, Bool.false_or, Bool.and_true, Bool.true_and]
  cases (value gate).decode <;> cases (value incoming).decode <;> simp
end SourceFP

/-- The source graph at N=1 retains the ordered zero/gate FMA. -/
theorem Graph.singleton_dseed (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word)
    (hn : c.N=1) (p j : Nat) (hp : p<c.programs) (hj : j<1024) :
    Graph.run c trees seed m .dseed p 0 j =
      .fma .zero (.word (m .gate (p/c.tiles) 0 0)) (.word (m .incoming p 0 j)) := by
  simp [Graph.run, Source.run_complete, Source.completeImage, hn, hp, hj,
    Reference.reverse, graphOps, inputsOf, Memory.map]

namespace NativeGraph
/-- The saved N=1 PTX has eight lane FMAs with ordered operands (gate,0,incoming).
All other cells retain their source graph. This is a local postlude lowering,
not a global commutation of the arbitrary ISA's FMA operation. -/
def run (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) : Memory Expr :=
  fun r p t j => if c.N=1 ∧ r=.dseed ∧ p<c.programs ∧ t=0 ∧ j<1024 then
    .fma (.word (m .gate (p/c.tiles) 0 0)) .zero (.word (m .incoming p 0 j))
    else Graph.run c trees seed m r p t j

def evalRun (isa : ISA) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) : Memory Word :=
  Memory.map (Expr.eval isa) (run c trees seed m)

theorem non_singleton (isa : ISA) (c : Shape) (trees : Trees c) (seed : Bool)
    (m : Memory Word) (hn : c.N≠1) :
    evalRun isa c trees seed m=Source.run (wordOps isa) c trees seed m := by
  rw [← Graph.eval_run isa]
  funext r p t j
  simp [evalRun,Memory.map,NativeGraph.run,hn]

/-- The sole reordered operation is bridged by a proved concrete software
identity. Native instruction conformance is handled separately below. -/
theorem eval_sourceFP (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) :
    evalRun SourceFP.scalar c trees seed m=Source.run (wordOps SourceFP.scalar) c trees seed m := by
  rw [← Graph.eval_run SourceFP.scalar]
  funext r p t j
  change (run c trees seed m r p t j).eval SourceFP.scalar =
    (Graph.run c trees seed m r p t j).eval SourceFP.scalar
  unfold run
  split_ifs with h
  · obtain ⟨hn,rfl,hp,rfl,hj⟩ := h
    rw [Graph.singleton_dseed c trees seed m hn p j hp hj]
    exact SourceFP.zero_fma_swap _ _
  · rfl
end NativeGraph

structure NumericDomain where
  add : Word → Word → Prop
  mul : Word → Word → Prop
  fma : Word → Word → Word → Prop

def Expr.Admitted (isa : ISA) (dom : NumericDomain) : Expr → Prop
  | .word _ | .zero => True
  | .add a b => a.Admitted isa dom ∧ b.Admitted isa dom ∧ dom.add (a.eval isa) (b.eval isa)
  | .mul a b => a.Admitted isa dom ∧ b.Admitted isa dom ∧ dom.mul (a.eval isa) (b.eval isa)
  | .fma a b c => a.Admitted isa dom ∧ b.Admitted isa dom ∧ c.Admitted isa dom ∧
      dom.fma (a.eval isa) (b.eval isa) (c.eval isa)

/-- Exactly three local scalar numerical laws. No kernel, dot, reduction,
result buffer, Real target, or postcondition is an argument to any law. -/
structure LocalConformance (isa : ISA) (dom : NumericDomain) : Prop where
  add : ∀ a b, dom.add a b → isa.add a b=SourceFP.scalar.add a b
  mul : ∀ a b, dom.mul a b → isa.mul a b=SourceFP.scalar.mul a b
  fma : ∀ a b c, dom.fma a b c → isa.fma a b c=SourceFP.scalar.fma a b c

theorem Expr.transport (isa : ISA) (dom : NumericDomain) (laws : LocalConformance isa dom)
    (e : Expr) (he : e.Admitted isa dom) : e.eval isa=e.eval SourceFP.scalar := by
  induction e with
  | word w => rfl
  | zero => rfl
  | add a b ha hb => rw [Expr.eval,laws.add _ _ he.2.2,ha he.1,hb he.2.1]; rfl
  | mul a b ha hb => rw [Expr.eval,laws.mul _ _ he.2.2,ha he.1,hb he.2.1]; rfl
  | fma a b c ha hb hc => rw [Expr.eval,laws.fma _ _ _ he.2.2.2,ha he.1,hb he.2.1,hc he.2.2.1]; rfl

/-- Every computed carry is present in O, DI or DL; every product/reduction
node is present in DG scratch or the host result. The graph includes every
complete tree leaf and all explicit zero seeds, including N=1 postlude FMAs. Both the source and lowered graphs are admitted,
so the two ordered N=1 triples are covered. This recursively checks operand
domains, not a desired output equation. -/
def GraphDomain (isa : ISA) (dom : NumericDomain) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) : Prop :=
  ∀ r p t j, (Graph.run c trees seed m r p t j).Admitted isa dom ∧
    (NativeGraph.run c trees seed m r p t j).Admitted isa dom

/-- Both ordered postlude operands follow from the public numerical domain. -/
theorem GraphDomain.singleton_orders (isa : ISA) (dom : NumericDomain) (c : Shape)
    (trees : Trees c) (seed : Bool) (m : Memory Word) (hd : GraphDomain isa dom c trees seed m)
    (hn : c.N=1) (p j : Nat) (hp : p<c.programs) (hj : j<1024) :
    dom.fma 0 (m .gate (p/c.tiles) 0 0) (m .incoming p 0 j) ∧
    dom.fma (m .gate (p/c.tiles) 0 0) 0 (m .incoming p 0 j) := by
  have h := hd .dseed p 0 j
  rw [Graph.singleton_dseed c trees seed m hn p j hp hj] at h
  simpa [NativeGraph.run,hn,hp,hj,Expr.Admitted,Expr.eval] using h

structure Numerical (isa : ISA) (dom : NumericDomain) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) : Prop where
  localLaws : LocalConformance isa dom
  operands : GraphDomain isa dom c trees seed m

def allOperands : NumericDomain := ⟨fun _ _ => True,fun _ _ => True,fun _ _ _ => True⟩
theorem Expr.all_admitted (isa : ISA) (e : Expr) : e.Admitted isa allOperands := by
  induction e <;> simp_all [Expr.Admitted,allOperands]

/-- Closed satisfiability witness for all words, dimensions, trees and flags.
This does not assert conformance of GPU hardware to the software profile. -/
theorem reference_witness (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) :
    Numerical SourceFP.scalar allOperands c trees seed m := by
  refine ⟨⟨fun _ _ _ => rfl,fun _ _ _ => rfl,fun _ _ _ _ => rfl⟩,?_⟩
  unfold GraphDomain
  intro r p t j
  exact ⟨Expr.all_admitted _ _, Expr.all_admitted _ _⟩

theorem native_complete (isa : ISA) (dom : NumericDomain) (c : Shape) (trees : Trees c)
    (seed : Bool) (m : Memory Word) (numeric : Numerical isa dom c trees seed m) :
    NativeGraph.evalRun isa c trees seed m=
      Source.completeImage (wordOps SourceFP.scalar) c trees seed m := by
  rw [← Source.run_complete]
  rw [← NativeGraph.eval_sourceFP]
  funext r p t j
  exact Expr.transport isa dom numeric.localLaws _ (numeric.operands r p t j).2

#specNonCircular LocalConformance avoiding [Graph.run,Source.run,Reference.forward]
#specNonCircular SourceFP.scalar avoiding [Graph.run,Source.run,Reference.forward]
#axiomsClean SourceFP.zero_fma_swap
#axiomsClean Graph.singleton_dseed
#axiomsClean NativeGraph.non_singleton
#axiomsClean NativeGraph.eval_sourceFP
#axiomsClean GraphDomain.singleton_orders
#axiomsClean Graph.eval_run
#axiomsClean reference_witness
#axiomsClean native_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
