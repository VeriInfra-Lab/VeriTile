import bench.tritonbench_g.triton_argmax.Proofs.AliasIR

/-! Structural extraction from finite, width-typed native scalar expressions.
The combiner retains its greater/equal/signed-index-less/and/or/select sites.
Every execution is computed by ScalarIR.Expr.eval, including native index
bits and sign extensions. Ordered tree agreement is derived by induction;
it is never a field of a runtime or compiler-semantics assumption. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.ReductionIR
open Source Reduction Emission SelectedEmission ScalarIR
variable {n : Nat} {graph : Graph n} {types : Ref → Kind}

structure Combiner (graph : Graph n) (types : Ref → Kind) (va vb ia ib : Ref) where
  g : Ref
  e : Ref
  l : Ref
  a : Ref
  p : Ref
  andOrder : Bool
  orOrder : Bool
  gType : types g=.predicate
  eType : types e=.predicate
  lType : types l=.predicate
  aType : types a=.predicate
  pType : types p=.predicate
  greater : graph.instruction g=some (.greater va vb)
  equal : graph.instruction e=some (.equal va vb)
  less : graph.instruction l=some (.indexLess ia ib)
  andEdge : graph.instruction a=some (if andOrder then .boolAnd e l else .boolAnd l e)
  orEdge : graph.instruction p=some (if orOrder then .boolOr g a else .boolOr a g)

def Combiner.expression (combiner : Combiner graph types va vb ia ib)
    (a : Expr valueBits graph types .value va) (b : Expr valueBits graph types .value vb)
    (i : Expr valueBits graph types (.index width) ia) (j : Expr valueBits graph types (.index width) ib) :
    Expr valueBits graph types .predicate combiner.p := by
  let greater := Expr.greater combiner.g a b combiner.gType combiner.greater
  let equal := Expr.equal combiner.e a b combiner.eType combiner.equal
  let less := Expr.less combiner.l i j combiner.lType combiner.less
  let conjunction : Expr valueBits graph types .predicate combiner.a :=
    if order : combiner.andOrder then
      Expr.both combiner.a equal less combiner.aType (by simpa only [order,if_true] using combiner.andEdge)
    else Expr.both combiner.a less equal combiner.aType (by simpa only [order,if_false] using combiner.andEdge)
  exact if order : combiner.orOrder then
    Expr.either combiner.p greater conjunction combiner.pType (by simpa only [order,if_true] using combiner.orEdge)
  else Expr.either combiner.p conjunction greater combiner.pType (by simpa only [order,if_false] using combiner.orEdge)

theorem Combiner.evaluated (combiner : Combiner graph types va vb ia ib)
    (a : Expr valueBits graph types .value va) (b : Expr valueBits graph types .value vb)
    (i : Expr valueBits graph types (.index width) ia) (j : Expr valueBits graph types (.index width) ib)
    (gt eq : α → α → Bool) (input : Fin n → α) :
    (combiner.expression a b i j).eval gt eq input=
      leftWins gt eq (a.eval gt eq input) (b.eval gt eq input)
        (i.eval gt eq input).toNat (j.eval gt eq input).toNat := by
  have indexComparison : decide ((i.eval gt eq input).toInt<(j.eval gt eq input).toInt)=
      decide ((i.eval gt eq input).toNat<(j.eval gt eq input).toNat) := by
    rw [index_signed_natural gt eq input i,index_signed_natural gt eq input j]
    congr 1
    exact propext Int.ofNat_lt
  cases first : combiner.andOrder <;> cases second : combiner.orOrder <;>
    simp [Combiner.expression,first,second,Expr.eval,
      indexComparison,leftWins,Bool.and_comm,Bool.or_comm]

inductive Pair (graph : Graph n) (types : Ref → Kind) : {width : Nat} → {v i : Ref} →
    Expr valueBits graph types .value v → Expr valueBits graph types (.index width) i → Tree n → Prop where
  | leaf (v i : Ref) (lane : Fin n) (width : Nat) (positive : 0<width)
      (fits : lane.val<2^(width-1)) (valueType : types v=.value) (indexType : types i=.index width)
      (valueInstruction : graph.instruction v=some (.input lane))
      (indexInstruction : graph.instruction i=some (.laneIndex lane)) :
      Pair graph types (.input v lane valueType valueInstruction)
        (.index i width positive lane fits indexType indexInstruction) (.leaf lane)
  | valueTransport {valueSite indexSite : Ref} {width : Nat} {tree : Tree n}
      {value : Expr valueBits graph types .value valueSite} {index : Expr valueBits graph types (.index width) indexSite}
      (source : Pair graph types value index tree) (site : Ref)
      (typed : types site=.value) (edge : NativeTransport.Checked graph valueBits site valueSite) :
      Pair graph types (.transport site value typed edge) index tree
  | indexTransport {valueSite indexSite : Ref} {width : Nat} {tree : Tree n}
      {value : Expr valueBits graph types .value valueSite} {index : Expr valueBits graph types (.index width) indexSite}
      (source : Pair graph types value index tree) (site : Ref)
      (typed : types site=.index width) (edge : NativeTransport.Checked graph width site indexSite) :
      Pair graph types value (.transport site index typed edge) tree
  | widen {valueSite indexSite : Ref} {width target : Nat} {tree : Tree n}
      {value : Expr valueBits graph types .value valueSite} {index : Expr valueBits graph types (.index width) indexSite}
      (source : Pair graph types value index tree) (site : Ref) (larger : width≤target)
      (typed : types site=.index target) (instruction : graph.instruction site=some (.copy indexSite))
      (earlier : graph.time indexSite<graph.time site) :
      Pair graph types value (.widen site index larger typed instruction earlier) tree
  | choose {va vb ia ib : Ref} {width : Nat} {leftTree rightTree : Tree n}
      {a : Expr valueBits graph types .value va} {b : Expr valueBits graph types .value vb}
      {i : Expr valueBits graph types (.index width) ia} {j : Expr valueBits graph types (.index width) ib}
      (left : Pair graph types a i leftTree) (right : Pair graph types b j rightTree)
      (combiner : Combiner graph types va vb ia ib) (valueSite indexSite : Ref)
      (valueType : types valueSite=.value) (indexType : types indexSite=.index width)
      (valueInstruction : graph.instruction valueSite=some (.selectValue combiner.p va vb))
      (indexInstruction : graph.instruction indexSite=some (.selectIndex combiner.p ia ib)) :
      Pair graph types
        (.selectValue valueSite (combiner.expression a b i j) a b valueType valueInstruction)
        (.selectIndex indexSite (combiner.expression a b i j) i j indexType indexInstruction)
        (.fork leftTree rightTree)

  | chooseAlias {va vb ia ib ga gb ja jb : Ref} {width compareWidth : Nat} {leftTree rightTree : Tree n}
      {a : Expr valueBits graph types .value va} {b : Expr valueBits graph types .value vb}
      {i : Expr valueBits graph types (.index width) ia} {j : Expr valueBits graph types (.index width) ib}
      {compareA : Expr valueBits graph types .value ga} {compareB : Expr valueBits graph types .value gb}
      {compareI : Expr valueBits graph types (.index compareWidth) ja} {compareJ : Expr valueBits graph types (.index compareWidth) jb}
      (left : Pair graph types a i leftTree) (right : Pair graph types b j rightTree)
      (aliasA : AliasIR.Value compareA a) (aliasB : AliasIR.Value compareB b)
      (aliasI : AliasIR.Index compareI i) (aliasJ : AliasIR.Index compareJ j)
      (combiner : Combiner graph types ga gb ja jb) (valueSite indexSite : Ref)
      (valueType : types valueSite=.value) (indexType : types indexSite=.index width)
      (valueInstruction : graph.instruction valueSite=some (.selectValue combiner.p va vb))
      (indexInstruction : graph.instruction indexSite=some (.selectIndex combiner.p ia ib)) :
      Pair graph types
        (.selectValue valueSite (combiner.expression compareA compareB compareI compareJ) a b valueType valueInstruction)
        (.selectIndex indexSite (combiner.expression compareA compareB compareI compareJ) i j indexType indexInstruction)
        (.fork leftTree rightTree)

theorem natural_of_signed (bits : BitVec width) (value : Nat) (signed : bits.toInt=Int.ofNat value) :
    bits.toNat=value := by
  have nonnegative : 0≤bits.toInt := by rw [signed]; exact Int.natCast_nonneg _
  have msb : bits.msb=Bool.false := by rw [BitVec.msb_eq_toInt]; simp [not_lt.mpr nonnegative]
  have equality := congrArg Int.toNat signed
  rw [BitVec.toNat_toInt_of_msb bits msb] at equality
  exact equality

theorem Pair.evaluated (derived : Pair graph types value index tree)
    (gt eq : α → α → Bool) (input : Fin n → α) :
    value.eval gt eq input=(tree.eval gt eq input).1 ∧
      (index.eval gt eq input).toNat=(tree.eval gt eq input).2.val := by
  induction derived with
  | leaf v i lane width positive fits valueType indexType valueInstruction indexInstruction =>
    exact ⟨rfl,natural_of_signed _ _ (index_literal_exact width lane.val positive fits)⟩
  | valueTransport source site typed edge ih => exact ih
  | indexTransport source site typed edge ih => exact ih
  | widen source site larger typed instruction earlier ih =>
    refine ⟨ih.1,?_⟩
    apply natural_of_signed
    simp only [Expr.eval,BitVec.toInt_signExtend_of_le larger,index_signed_natural,ih.2]
  | @choose va vb ia ib width leftTree rightTree a b i j left right combiner valueSite indexSite valueType indexType valueInstruction indexInstruction ihLeft ihRight =>
    have condition := combiner.evaluated a b i j gt eq input
    rw [ihLeft.1,ihLeft.2,ihRight.1,ihRight.2] at condition
    constructor
    · simp only [Expr.eval,condition,ihLeft.1,ihRight.1]
      exact congrArg Prod.fst (choose_tree gt eq input _ _)
    · simp only [Expr.eval,condition,apply_ite,ihLeft.2,ihRight.2]
      have selected := congrArg (fun result : α × Fin n => result.2.val)
        (choose_tree gt eq input leftTree rightTree)
      simpa only [choosePair,leftWins,apply_ite] using selected

  | @chooseAlias va vb ia ib ga gb ja jb width compareWidth leftTree rightTree a b i j compareA compareB compareI compareJ left right aliasA aliasB aliasI aliasJ combiner valueSite indexSite valueType indexType valueInstruction indexInstruction ihLeft ihRight =>
    have condition := combiner.evaluated compareA compareB compareI compareJ gt eq input
    rw [aliasA.evaluated gt eq input,aliasB.evaluated gt eq input,
      aliasI.evaluated gt eq input,aliasJ.evaluated gt eq input,
      ihLeft.1,ihLeft.2,ihRight.1,ihRight.2] at condition
    constructor
    · simp only [Expr.eval,condition,ihLeft.1,ihRight.1]
      exact congrArg Prod.fst (choose_tree gt eq input _ _)
    · simp only [Expr.eval,condition,apply_ite,ihLeft.2,ihRight.2]
      have selected := congrArg (fun result : α × Fin n => result.2.val)
        (choose_tree gt eq input leftTree rightTree)
      simpa only [choosePair,leftWins,apply_ite] using selected

structure Program (valueBits n : Nat) where
  raw : RawArtifact n
  code : ScalarIR.Code valueBits raw.graph
  indexWidth : Nat
  value : Expr valueBits raw.graph code.types .value raw.rootValue
  index : Expr valueBits raw.graph code.types (.index indexWidth) raw.rootIndex
  tree : Tree n
  extraction : Pair raw.graph code.types value index tree
  covers : tree.Complete

def Combiner.source (combiner : Combiner graph types va vb ia ib) :
    PredicateEdges graph combiner.p va vb ia ib := by
  have andEdge : graph.instruction combiner.a=some (.boolAnd combiner.e combiner.l) ∨
      graph.instruction combiner.a=some (.boolAnd combiner.l combiner.e) := by
    cases order : combiner.andOrder
    · exact Or.inr (by simpa only [order,Bool.false_eq_true,if_false] using combiner.andEdge)
    · exact Or.inl (by simpa only [order,if_true] using combiner.andEdge)
  have orEdge : graph.instruction combiner.p=some (.boolOr combiner.g combiner.a) ∨
      graph.instruction combiner.p=some (.boolOr combiner.a combiner.g) := by
    cases order : combiner.orOrder
    · exact Or.inr (by simpa only [order,Bool.false_eq_true,if_false] using combiner.orEdge)
    · exact Or.inl (by simpa only [order,if_true] using combiner.orEdge)
  exact ⟨combiner.g,combiner.e,combiner.l,combiner.a,combiner.greater,combiner.equal,combiner.less,andEdge,orEdge⟩

theorem Pair.source {v i : Ref} {width : Nat}
    {value : Expr valueBits graph types .value v} {index : Expr valueBits graph types (.index width) i}
    (derived : Pair graph types value index tree) : PairDerived graph v i tree := by
  induction derived with
  | leaf v i lane width positive fits valueType indexType valueInstruction indexInstruction =>
    exact .leaf valueInstruction indexInstruction
  | valueTransport source site typed edge ih => exact .valueTransport edge.source ih
  | indexTransport source site typed edge ih => exact .indexTransport edge.source ih
  | widen source site larger typed instruction earlier ih => exact .indexTransport (.copy instruction earlier) ih
  | choose left right combiner valueSite indexSite valueType indexType valueInstruction indexInstruction ihLeft ihRight =>
    exact .choose ihLeft ihRight combiner.source valueInstruction indexInstruction
  | chooseAlias left right aliasA aliasB aliasI aliasJ combiner valueSite indexSite valueType indexType valueInstruction indexInstruction ihLeft ihRight =>
    exact .chooseAlias ihLeft ihRight combiner.source aliasA.source aliasB.source aliasI.source aliasJ.source
      valueInstruction indexInstruction

def Program.source (program : Program valueBits n) : Extraction program.raw :=
  ⟨program.tree,program.extraction.source,program.covers⟩

def Program.nativeRun (program : Program valueBits n)
    (gt eq : BitVec valueBits → BitVec valueBits → Bool) (input : Fin n → BitVec valueBits) :
    BitVec valueBits × BitVec program.indexWidth :=
  (program.value.nativeEval gt eq input,program.index.nativeEval gt eq input)

def Program.run (program : Program valueBits n) (gt eq : α → α → Bool) (input : Fin n → α) : α × BitVec program.indexWidth :=
  (program.value.eval gt eq input,program.index.eval gt eq input)

theorem Program.evaluated (program : Program valueBits n) (gt eq : α → α → Bool) (input : Fin n → α) :
    (program.run gt eq input).1=(program.tree.eval gt eq input).1 ∧
      (program.run gt eq input).2.toNat=(program.tree.eval gt eq input).2.val :=
  program.extraction.evaluated gt eq input

theorem Program.native_evaluated (program : Program valueBits n)
    (gt eq : BitVec valueBits → BitVec valueBits → Bool) (input : Fin n → BitVec valueBits) :
    (program.nativeRun gt eq input).1=(program.tree.eval gt eq input).1 ∧
      (program.nativeRun gt eq input).2.toNat=(program.tree.eval gt eq input).2.val := by
  have value := program.value.native_erases gt eq input
  have index := program.index.native_erases gt eq input
  change program.value.nativeEval gt eq input=program.value.eval gt eq input at value
  change program.index.nativeEval gt eq input=program.index.eval gt eq input at index
  simpa only [nativeRun,value,index] using program.evaluated gt eq input

#axiomsClean Program.native_evaluated
#axiomsClean Combiner.evaluated
#axiomsClean Combiner.source
#axiomsClean natural_of_signed
#axiomsClean Pair.evaluated
#axiomsClean Pair.source
#axiomsClean Program.evaluated
end VeriTile.Bench.Repairs.TritonArgmax.Revision004.ReductionIR
