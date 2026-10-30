import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Floating

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source
noncomputable section
set_option maxHeartbeats 6000000

/-- Lowered scalar instructions. FMA is explicit and receives unrounded
operands. Host operations use separate scalar ports; every reduction expands its explicit seeded binary tree. -/
inductive Instruction where
  | raw (bits : Word)
  | literal (value : ℝ)
  | natural (value : Nat)
  | negativeInfinity
  | binary (operation : Binary32) (left right : Instruction)
  | unary (operation : Unary32) (value : Instruction)
  | fma (left right addend : Instruction)
  | select (condition : Bool) (yes no : Instruction)
  | negation (value : Instruction)
  | sum (width : Nat) (values : Fin width → Instruction)
  | maximum (width : Nat) (values : Fin width → Instruction)
  | hostAdd (left right : Instruction)
  | hostMultiply (left right : Instruction)
  | hostSquare (value : Instruction)
  | hostScale (scale : ℝ) (value : Instruction)
  | hostSum (width : Nat) (values : Fin width → Instruction)
  | hostLSE (width : Nat) (values : Fin width → Instruction)
  | rankSum (width : Nat) (values : Fin width → Instruction)

def runInstruction (n : Ports) (profile : Profile) : Instruction → Word
  | .raw x => x
  | .literal x => n.literal x
  | .natural x => n.natural x
  | .negativeInfinity => n.negativeInfinity
  | .binary op a b =>
    match op with
    | .add => n.add (runInstruction n profile a) (runInstruction n profile b)
    | .sub => n.sub (runInstruction n profile a) (runInstruction n profile b)
    | .mul => n.mul (runInstruction n profile a) (runInstruction n profile b)
    | .div => n.div (runInstruction n profile a) (runInstruction n profile b)
  | .unary op x =>
    match op with
    | .exp => n.exp (runInstruction n profile x)
    | .log => n.log (runInstruction n profile x)
  | .fma a b c => n.fma (runInstruction n profile a) (runInstruction n profile b) (runInstruction n profile c)
  | .select condition yes no => if condition then runInstruction n profile yes else runInstruction n profile no
  | .negation x => n.neg (runInstruction n profile x)
  | .sum _ xs => reduce profile.sumTree n.add (n.literal 0) (List.ofFn (fun i => runInstruction n profile (xs i)))
  | .maximum _ xs => reduce profile.maxTree n.max n.negativeInfinity (List.ofFn (fun i => runInstruction n profile (xs i)))
  | .hostAdd a b => n.hostAdd (runInstruction n profile a) (runInstruction n profile b)
  | .hostMultiply a b => n.hostMul (runInstruction n profile a) (runInstruction n profile b)
  | .hostSquare x => let a := runInstruction n profile x; n.hostMul a a
  | .hostScale scale x => n.hostMul (n.literal scale) (runInstruction n profile x)
  | .hostSum _ xs => reduce profile.hostSumTree n.hostAdd (n.literal 0) (List.ofFn (fun i => runInstruction n profile (xs i)))
  | .hostLSE _ xs => hostLSE n profile (List.ofFn (fun i => runInstruction n profile (xs i)))
  | .rankSum _ xs => reduce profile.hostSumTree n.hostAdd (n.literal 0) (List.ofFn (fun i => runInstruction n profile (xs i)))

structure Compiled where
  value : Instruction
  product : Option (Instruction × Instruction) := none

def Compiled.realize (n : Ports) (profile : Profile) (c : Compiled) : Evaluated :=
  ⟨runInstruction n profile c.value,c.product.map (fun xy => (runInstruction n profile xy.1,runInstruction n profile xy.2))⟩

def emitAdd (a b : Compiled) : Compiled :=
  match b.product with
  | some (x,y) => ⟨.fma x y a.value,none⟩
  | none => match a.product with
    | some (x,y) => ⟨.fma x y b.value,none⟩
    | none => ⟨.binary .add a.value b.value,none⟩

def emitSub (a b : Compiled) : Compiled :=
  match b.product with
  | some (x,y) => ⟨.fma (.negation x) y a.value,none⟩
  | none => ⟨.binary .sub a.value b.value,none⟩

def compile : Expr → Compiled
  | .raw x => ⟨.raw x,none⟩
  | .literal x => ⟨.literal x,none⟩
  | .natural x => ⟨.natural x,none⟩
  | .negativeInfinity => ⟨.negativeInfinity,none⟩
  | .barrier x => ⟨(compile x).value,none⟩
  | .binary op a b =>
    let left := compile a
    let right := compile b
    match op with
    | .add => emitAdd left right
    | .sub => emitSub left right
    | .mul => ⟨.binary .mul left.value right.value,some (left.value,right.value)⟩
    | .div => ⟨.binary .div left.value right.value,none⟩
  | .unary op x => ⟨.unary op (compile x).value,none⟩
  | .sum width xs => ⟨.sum width (fun i => (compile (xs i)).value),none⟩
  | .maximum width xs => ⟨.maximum width (fun i => (compile (xs i)).value),none⟩
  | .negation x => ⟨.negation (compile x).value,none⟩
  | .hostAdd a b => ⟨.hostAdd (compile a).value (compile b).value,none⟩
  | .hostMultiply a b => ⟨.hostMultiply (compile a).value (compile b).value,none⟩
  | .hostSquare x => ⟨.hostSquare (compile x).value,none⟩
  | .hostScale scale x => ⟨.hostScale scale (compile x).value,none⟩
  | .hostSum width xs => ⟨.hostSum width (fun i => (compile (xs i)).value),none⟩
  | .hostLSE width xs => ⟨.hostLSE width (fun i => (compile (xs i)).value),none⟩
  | .rankSum width xs => ⟨.rankSum width (fun i => (compile (xs i)).value),none⟩

theorem emit_add_local (n : Ports) (profile : Profile) (a b : Compiled) :
    (emitAdd a b).realize n profile=add n (a.realize n profile) (b.realize n profile) := by
  rcases a with ⟨av,ap⟩
  rcases b with ⟨bv,bp⟩
  cases bp with
  | some pair => rcases pair with ⟨x,y⟩; rfl
  | none =>
    cases ap with
    | some pair => rcases pair with ⟨x,y⟩; rfl
    | none => rfl

theorem emit_sub_local (n : Ports) (profile : Profile) (a b : Compiled) :
    (emitSub a b).realize n profile=sub n (a.realize n profile) (b.realize n profile) := by
  rcases b with ⟨bv,bp⟩
  cases bp with
  | some pair => rcases pair with ⟨x,y⟩; rfl
  | none => rfl

/-- The local lowering proof is structural; there is no whole-expression or
source-output equivalence premise. Compiler peephole rewrites are kept as
separate instruction-sized obligations rather than silently changing fusion. -/
theorem compile_realizes (n : Ports) (profile : Profile) (expression : Expr) :
    (compile expression).realize n profile=evaluate n profile expression := by
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
    | add => simpa only [compile,evaluate] using (emit_add_local n profile (compile a) (compile b)).trans (congrArg₂ (add n) ia ib)
    | sub => simpa only [compile,evaluate] using (emit_sub_local n profile (compile a) (compile b)).trans (congrArg₂ (sub n) ia ib)
    | mul =>
      have va := congrArg Evaluated.value ia
      have vb := congrArg Evaluated.value ib
      simp only [Compiled.realize] at va vb
      simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_some,va,vb]
    | div =>
      have va := congrArg Evaluated.value ia
      have vb := congrArg Evaluated.value ib
      simp only [Compiled.realize] at va vb
      simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_none,va,vb]
  | unary op x ih =>
    have v := congrArg Evaluated.value ih
    simp only [Compiled.realize] at v
    cases op <;> simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_none,v]
  | sum width xs ih | maximum width xs ih | hostSum width xs ih | hostLSE width xs ih | rankSum width xs ih =>
    have values : (fun i => runInstruction n profile (compile (xs i)).value)=(fun i => (evaluate n profile (xs i)).value) :=
      funext (fun i => congrArg Evaluated.value (ih i))
    simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_none,values]
  | negation x ih | hostSquare x ih =>
    have v := congrArg Evaluated.value ih
    simp only [Compiled.realize] at v
    simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_none,v]
  | hostScale scale x ih =>
    have v := congrArg Evaluated.value ih
    simp only [Compiled.realize] at v
    simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_none,v]
  | hostAdd a b ia ib | hostMultiply a b ia ib =>
    have va := congrArg Evaluated.value ia
    have vb := congrArg Evaluated.value ib
    simp only [Compiled.realize] at va vb
    simp only [compile,Compiled.realize,runInstruction,evaluate,Option.map_none,va,vb]

theorem compiled_word (n : Ports) (profile : Profile) (expression : Expr) :
    runInstruction n profile (compile expression).value=word n profile expression := congrArg Evaluated.value (compile_realizes n profile expression)

/-- Native code generation may commute a binary multiplication/FMA, use add
for multiplication by two, or add negative one for subtracting one. These
ports are only scalar instruction identities; NaN and signed-zero behavior is
part of each explicit all-pattern obligation. -/
structure Peepholes (n : Ports) : Prop where
  addComm : ∀ x y, n.add x y=n.add y x
  mulComm : ∀ x y, n.mul x y=n.mul y x
  fmaComm : ∀ x y z, n.fma x y z=n.fma y x z
  negFmaExchange : ∀ x y z, n.fma (n.neg x) y z=n.fma (n.neg y) x z
  twice : ∀ x, n.mul (n.literal 2) x=n.add x x
  minusOne : ∀ x, n.sub x (n.literal 1)=n.add x (n.literal (-1))

theorem doubled_scale_local (n : Ports) (profile : Profile) (laws : Peepholes n) (scale : Word) :
    runInstruction n profile (.binary .mul (.literal 2) (.raw scale))=
      runInstruction n profile (.binary .add (.raw scale) (.raw scale)) := laws.twice scale

theorem exchanged_fma_local (n : Ports) (profile : Profile) (laws : Peepholes n) (a b c : Word) :
    runInstruction n profile (.fma (.raw a) (.raw b) (.raw c))=
      runInstruction n profile (.fma (.raw b) (.raw a) (.raw c)) := laws.fmaComm a b c

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
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
