import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Floating

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open Source
noncomputable section
set_option maxHeartbeats 6000000

/-- Fully lowered instructions contain only fixed-arity scalar operations
and explicit one-word transfers. Reduction constructors do not exist here. -/
abbrev Instruction := ScalarGraph 0

def runInstruction (n : Numerics) (instruction : Instruction) : Word :=
  instruction.eval n Fin.elim0

structure Compiled where
  value : Instruction
  product : Option (Instruction × Instruction) := none

def Compiled.realize (n : Numerics) (c : Compiled) : Evaluated :=
  ⟨runInstruction n c.value,c.product.map (fun xy => (runInstruction n xy.1,runInstruction n xy.2))⟩

def emitAdd (a b : Compiled) : Compiled :=
  match b.product with
  | some (x,y) => ⟨.fma x y a.value,none⟩
  | none => match a.product with
    | some (x,y) => ⟨.fma x y b.value,none⟩
    | none => ⟨.binary .add a.value b.value,none⟩

def emitSub (a b : Compiled) : Compiled :=
  match b.product with
  | some (x,y) => ⟨.fma (.unary .neg x) y a.value,none⟩
  | none => ⟨.binary .sub a.value b.value,none⟩

def compile (p : ReductionProfile) : Expr → Compiled
  | .raw x => ⟨.raw x,none⟩
  | .literal x => ⟨.literal x,none⟩
  | .natural x => ⟨.natural x,none⟩
  | .negativeInfinity => ⟨.negativeInfinity,none⟩
  | .barrier x => ⟨(compile p x).value,none⟩
  | .binary op a b =>
    let left := compile p a
    let right := compile p b
    match op with
    | .add => emitAdd left right
    | .sub => emitSub left right
    | .mul => ⟨.binary .mul left.value right.value,some (left.value,right.value)⟩
    | .div => ⟨.binary .div left.value right.value,none⟩
  | .unary op x => ⟨.unary (match op with | .exp => .exp | .log => .log) (compile p x).value,none⟩
  | .sum width xs => ⟨(p.sumGraph width).bind (fun i => (compile p (xs i)).value),none⟩
  | .maximum width xs => ⟨(p.maximumGraph width).bind (fun i => (compile p (xs i)).value),none⟩
  | .negation x => ⟨.unary .neg (compile p x).value,none⟩
  | .hostAdd a b => ⟨.binary .hostAdd (compile p a).value (compile p b).value,none⟩
  | .hostSquare x => ⟨.unary .hostSquare (compile p x).value,none⟩
  | .hostScale scale x => ⟨.hostScale scale (compile p x).value,none⟩
  | .hostSum width xs => ⟨(p.hostSumGraph width).bind (fun i => (compile p (xs i)).value),none⟩
  | .hostLSE width xs => ⟨(p.hostLSEGraph width).bind (fun i => (compile p (xs i)).value),none⟩
  | .rankSum width xs => ⟨(p.rankSumGraph width).bind (fun i => (compile p (xs i)).value),none⟩

theorem emit_add_local (n : Numerics) (a b : Compiled) :
    (emitAdd a b).realize n=add n (a.realize n) (b.realize n) := by
  rcases a with ⟨av,ap⟩
  rcases b with ⟨bv,bp⟩
  cases bp with
  | some pair => rcases pair with ⟨x,y⟩; rfl
  | none =>
    cases ap with
    | some pair => rcases pair with ⟨x,y⟩; rfl
    | none => rfl

theorem emit_sub_local (n : Numerics) (a b : Compiled) :
    (emitSub a b).realize n=sub n (a.realize n) (b.realize n) := by
  rcases b with ⟨bv,bp⟩
  cases bp with
  | some pair => rcases pair with ⟨x,y⟩; rfl
  | none => rfl

/-- The local lowering proof is structural; there is no whole-expression or
source-output equivalence premise. Compiler peephole rewrites are kept as
separate instruction-sized obligations rather than silently changing fusion. -/
theorem compile_realizes (p : ReductionProfile) (n : Numerics) (expression : Expr) :
    (compile p expression).realize n=evaluate p n expression := by
  induction expression with
  | raw => rfl
  | literal => rfl
  | natural => rfl
  | negativeInfinity => rfl
  | barrier x ih =>
    have value := congrArg Evaluated.value ih
    simp only [Compiled.realize] at value
    simpa only [compile,Compiled.realize,evaluate,Option.map_none] using congrArg (fun v => Evaluated.mk v none) value
  | binary op a b ia ib =>
    cases op with
    | add => simpa only [compile,evaluate] using (emit_add_local n (compile p a) (compile p b)).trans (congrArg₂ (add n) ia ib)
    | sub => simpa only [compile,evaluate] using (emit_sub_local n (compile p a) (compile p b)).trans (congrArg₂ (sub n) ia ib)
    | mul =>
      have va := congrArg Evaluated.value ia
      have vb := congrArg Evaluated.value ib
      simp only [Compiled.realize,runInstruction] at va vb
      simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval,ScalarBinary.run,ScalarUnary.run,evaluate,Option.map_some,va,vb]
    | div =>
      have va := congrArg Evaluated.value ia
      have vb := congrArg Evaluated.value ib
      simp only [Compiled.realize,runInstruction] at va vb
      simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval,ScalarBinary.run,ScalarUnary.run,evaluate,Option.map_none,va,vb]
  | unary op x ih =>
    have v := congrArg Evaluated.value ih
    simp only [Compiled.realize,runInstruction] at v
    cases op <;> simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval,ScalarBinary.run,ScalarUnary.run,evaluate,Option.map_none,v]
  | sum width xs ih | maximum width xs ih | hostSum width xs ih | hostLSE width xs ih | rankSum width xs ih =>
    have values : (fun i => ScalarGraph.eval n Fin.elim0 (compile p (xs i)).value)=(fun i => (evaluate p n (xs i)).value) :=
      funext (fun i => congrArg Evaluated.value (ih i))
    simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval_bind,evaluate,Option.map_none]
    rw [values]
  | negation x ih | hostSquare x ih =>
    have v := congrArg Evaluated.value ih
    simp only [Compiled.realize,runInstruction] at v
    simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval,ScalarBinary.run,ScalarUnary.run,evaluate,Option.map_none,v]
  | hostScale scale x ih =>
    have v := congrArg Evaluated.value ih
    simp only [Compiled.realize,runInstruction] at v
    simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval,ScalarBinary.run,ScalarUnary.run,evaluate,Option.map_none,v]
  | hostAdd a b ia ib =>
    have va := congrArg Evaluated.value ia
    have vb := congrArg Evaluated.value ib
    simp only [Compiled.realize,runInstruction] at va vb
    simp only [compile,Compiled.realize,runInstruction,ScalarGraph.eval,ScalarBinary.run,ScalarUnary.run,evaluate,Option.map_none,va,vb]

theorem compiled_word (p : ReductionProfile) (n : Numerics) (expression : Expr) :
    runInstruction n (compile p expression).value=word p n expression := congrArg Evaluated.value (compile_realizes p n expression)

/-- Native code generation may commute a binary multiplication/FMA, use add
for multiplication by two, or add negative one for subtracting one. These
ports are only scalar instruction identities; NaN and signed-zero behavior is
part of each explicit all-pattern obligation. -/
structure Peepholes (n : Numerics) : Prop where
  addComm : ∀ x y, n.add x y=n.add y x
  mulComm : ∀ x y, n.mul x y=n.mul y x
  fmaComm : ∀ x y z, n.fma x y z=n.fma y x z
  twice : ∀ x, n.mul (n.literal 2) x=n.add x x
  minusOne : ∀ x, n.sub x (n.literal 1)=n.add x (n.literal (-1))

theorem doubled_scale_local (n : Numerics) (laws : Peepholes n) (scale : Word) :
    runInstruction n (.binary .mul (.literal 2) (.raw scale))=
      runInstruction n (.binary .add (.raw scale) (.raw scale)) := laws.twice scale

theorem exchanged_fma_local (n : Numerics) (laws : Peepholes n) (a b c : Word) :
    runInstruction n (.fma (.raw a) (.raw b) (.raw c))=
      runInstruction n (.fma (.raw b) (.raw a) (.raw c)) := laws.fmaComm a b c

#axiomsClean emit_add_local
#axiomsClean emit_sub_local
#axiomsClean compile_realizes
#axiomsClean compiled_word
#axiomsClean doubled_scale_local
#axiomsClean exchanged_fma_local
#specNonCircular Peepholes avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
