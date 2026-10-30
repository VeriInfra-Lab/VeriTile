import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def SupportedBinary (op : Binary) : Prop :=
  op=.add ∨ op=.sub ∨ op=.mul ∨ op=.div ∨ op=.max

/-- Exactly the instruction vocabulary evaluated by the public nine ports.
Integer-to-float conversions and other generic Algebra operations are absent. -/
def Good : Expr → Prop
  | .leaf _ | .zero | .negInf | .sentinel | .literal .. => True
  | .fromNat .. | .fromInt .. => False
  | .cast _ source target x => source=target ∧ Good x
  | .unary _ op x => op=.exp2 ∧ Good x
  | .binary _ op x y => SupportedBinary op ∧ Good x ∧ Good y
  | .product _ x y => Good x ∧ Good y
  | .fma _ x y z => Good x ∧ Good y ∧ Good z
  | .reduction _ op _ _ xs => (op=.max ∨ op=.add) ∧ ∀ i,Good (xs i)
  | .dotAcc _ left right out _ _ _ _ _ a b acc _ =>
    left=.fp32 ∧ right=.fp32 ∧ out=.fp32 ∧ (∀ i,Good (a i)) ∧ (∀ i,Good (b i)) ∧ Good acc

@[simp] theorem good_literal (p : Option ComputeDType) (x : ℝ) : Good (literal p x) := by
  unfold literal
  split_ifs <;> trivial

theorem good_consume (p : Option ComputeDType) (op : Binary) (x y : Expr)
    (hop : SupportedBinary op) (hx : Good x) (hy : Good y) : Good (consume p op x y) := by
  rcases hop with rfl|rfl|rfl|rfl|rfl
  · cases x <;> cases y <;> simp_all [consume,Good,SupportedBinary]
  · exact ⟨by simp [SupportedBinary],hx,hy⟩
  · exact ⟨hx,hy⟩
  · exact ⟨by simp [SupportedBinary],hx,hy⟩
  · exact ⟨by simp [SupportedBinary],hx,hy⟩

@[simp] theorem good_cast_real (p : Option ComputeDType) (x : Expr) : Good (cast p .real .real x) ↔ Good x := by
  simp [cast,nativeDType]

theorem good_reduction (plan : Plan) (p : Option ComputeDType) (isMax : Bool)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : TileIndex shape → Expr)
    (hx : ∀ i,Good (xs i)) (i : TileIndex (TileShape.reduceShape shape ax keep)) :
    Good (reduction plan p isMax ax keep xs i) := by
  unfold reduction Good
  constructor
  · cases isMax <;> simp
  · intro j
    cases keep <;> exact hx _

def GoodMemory (s : State Expr) := ∀ rg off,Good ((s.mem rg off).read .real)
def GoodReg (s : State Expr) (shape : TileShape) (name : RegName) : Prop :=
  ∀ v,s.regs .real shape name=some v → ∀ i,Good (v i)
def Live (p : Layout) (s : State Expr) :=
  GoodReg s [p.BM,p.D] "q" ∧ GoodReg s [p.BM] "m_i" ∧
  GoodReg s [p.BM] "l_i" ∧ GoodReg s [p.BM,p.D] "acc"

theorem prelude_good (p : Layout) (r : Regions) (scale : ℝ) (plan : Plan)
    (s : State Expr) (hm : GoodMemory s) :
    ∃ t,run ((backend plan).withDefaultPrecision .fp32) (FPSuccess.parts p r scale).before s=some t ∧
      Live p t ∧ GoodMemory t := by
  simp (config := {maxSteps := 500000}) [FPSuccess.parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Live,GoodReg,GoodMemory,backend,FPSuccess.TotalDots.install,
    dots,scalar,cast,nativeDType,consume,Good,literal,bop]
  constructor
  · intro i
    split_ifs <;> intro d
    all_goals first | exact hm _ _ | trivial
  · exact hm

theorem body_good (p : Layout) (r : Regions) (scale : ℝ) (plan : Plan)
    (hBN : 0<p.BN) (s : State Expr) (hs : FPSuccess.Ready p s) (hl : Live p s)
    (hm : GoodMemory s) (cur : Int) :
    ∃ t,run ((backend plan).withDefaultPrecision .fp32) (FPSuccess.parts p r scale).body
      (s.setReg "start_n" .int [] (fun _ => cur))=some t ∧ Live p t ∧ GoodMemory t := by
  rcases hs with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases Option.isSome_iff_exists.mp h0 with ⟨v0,h0⟩
  rcases Option.isSome_iff_exists.mp h1 with ⟨v1,h1⟩
  rcases Option.isSome_iff_exists.mp h2 with ⟨v2,h2⟩
  rcases Option.isSome_iff_exists.mp h3 with ⟨v3,h3⟩
  rcases Option.isSome_iff_exists.mp h4 with ⟨v4,h4⟩
  rcases Option.isSome_iff_exists.mp h5 with ⟨v5,h5⟩
  rcases Option.isSome_iff_exists.mp h6 with ⟨v6,h6⟩
  rcases Option.isSome_iff_exists.mp h7 with ⟨v7,h7⟩
  rcases Option.isSome_iff_exists.mp h8 with ⟨v8,h8⟩
  rcases Option.isSome_iff_exists.mp h9 with ⟨v9,h9⟩
  rcases Option.isSome_iff_exists.mp h10 with ⟨v10,h10⟩
  rcases Option.isSome_iff_exists.mp h11 with ⟨v11,h11⟩
  rcases Option.isSome_iff_exists.mp h12 with ⟨v12,h12⟩
  rcases Option.isSome_iff_exists.mp h13 with ⟨v13,h13⟩
  rcases Option.isSome_iff_exists.mp h14 with ⟨v14,h14⟩
  have gq := hl.1 v7 h7
  have gm := hl.2.1 v8 h8
  have gl := hl.2.2.1 v9 h9
  have ga := hl.2.2.2 v10 h10
  unfold GoodMemory at hm
  simp (config := {maxSteps := 1000000}) [FPSuccess.parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,FPSuccess.Ready,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    TileShape.axisDim,hBN,backend,FPSuccess.TotalDots.install,dots,scalar,
    Live,GoodReg,GoodMemory,cast,nativeDType,Good,consume,SupportedBinary,gq,gm,gl,ga,hm,
    reduction,reductionInputs,literal,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  simp (config := {maxSteps := 1000000}) [apply_ite Good,Good,gq,gm,gl,ga,hm]


end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
