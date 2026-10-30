import bench.tritonbench_g.triton_attention.Proofs.FPBackend
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
structure PreParameters where
  dtype : FloatDType
  (output gradient denominator newGradient delta : RegionName)
  (rows head : Nat)
abbrev PreRow (p : PreParameters) := TileIndex [p.rows]
abbrev PreLane (p : PreParameters) := TileIndex [p.rows,p.head]
def preSource (p : PreParameters) :=
  preprocessTyped p.dtype .fp32 p.output p.gradient p.denominator p.newGradient p.delta p.rows p.head
def preRow (p : PreParameters) (s : State α) (i : PreRow p) := s.pids 0*p.rows+i.1.val
def preOffset (p : PreParameters) (s : State α) (i : PreLane p) :=
  (s.pids 0*p.rows+i.1.val)*p.head+i.2.1.val
def preInput (P : Primitives α) (p : PreParameters) (s : State α) (r : RegionName) (i : PreLane p) : α :=
  P.cast (some .fp32) p.dtype .real (toFloat p.dtype ((s.mem r (preOffset p s i)).read p.dtype.toTileDType))
def preDenominator (P : Primitives α) (p : PreParameters) (s : State α) (i : PreRow p) : α :=
  P.cast (some .fp32) .fp32 .real ((s.mem p.denominator (preRow p s i)).read .fp32)
def preQuotient (P : Primitives α) (p : PreParameters) (s : State α) (i : PreLane p) : α :=
  P.binary (some .fp32) .real .div (preInput P p s p.gradient i) (preDenominator P p s (i.1,PUnit.unit))
def preProducts (P : Primitives α) (p : PreParameters) (s : State α) (i : PreLane p) : α :=
  P.binary (some .fp32) .real .mul (preInput P p s p.output i) (preQuotient P p s i)
def preDelta (P : Primitives α) (p : PreParameters) (s : State α) : PreRow p → α :=
  P.sum (some .fp32) ⟨1,by simp⟩ Bool.false (preProducts P p s)
def preGradientCell (P : Primitives α) (p : PreParameters) (s : State α) (i : PreLane p) : Cell α :=
  .mk p.dtype.toTileDType (ofFloat p.dtype (P.cast none .real p.dtype (preQuotient P p s i)))
def preDeltaCell (P : Primitives α) (p : PreParameters) (s : State α) (i : PreRow p) : Cell α :=
  .mk .fp32 (P.cast none .real .fp32 (preDelta P p s i))
def preWrite (P : Primitives α) (p : PreParameters) (initial current : State α) : State α :=
  let t := (TileShape.allIndices [p.rows,p.head]).foldl (fun t i =>
    t.write p.newGradient (preOffset p initial i) (preGradientCell P p initial i)) current
  (TileShape.allIndices [p.rows]).foldl (fun t i =>
    t.write p.delta (preRow p initial i) (preDeltaCell P p initial i)) t
def preRegisters (P : Primitives α) (p : PreParameters) (s : State α) : State α :=
  let t := s.setReg "off_m" .nat [p.rows] (preRow p s)
  let t := t.setReg "off_n" .nat [p.head] (fun i => i.1.val)
  let t := t.setReg "o" .real [p.rows,p.head] (preInput P p s p.output)
  let t := t.setReg "do_val" .real [p.rows,p.head] (preInput P p s p.gradient)
  let t := t.setReg "denom" .real [p.rows] (preDenominator P p s)
  let t := t.setReg "do_val" .real [p.rows,p.head] (preQuotient P p s)
  t.setReg "delta" .real [p.rows] (preDelta P p s)

theorem preprocessExecution (rest : Algebra α) (P : Primitives α) (p : PreParameters) (s : State α) :
    FP.Structural.exec (backend rest P) (preSource p) s=
      some (preWrite P p s (preRegisters P p s)) := by
  simp [FP.Structural.exec,preSource,preprocessTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,store,backend,preWrite,preRegisters,preRow,preOffset,
    preInput,preDenominator,preQuotient,preProducts,preDelta,preGradientCell,preDeltaCell,
    Region.cast,FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  rfl

theorem scatterMemory {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (l : List ι) (initial current : State α) (h : current.mem=initial.mem) :
    (l.foldl (fun t i => t.write r (off i) (v i)) current).mem =
      (l.foldl (fun t i => t.write r (off i) (v i)) initial).mem := by
  induction l generalizing current initial with
  | nil => exact h
  | cons i l ih =>
    simp only [List.foldl_cons]
    apply ih
    simp [State.write,h]

theorem preWrite_memory (P : Primitives α) (p : PreParameters) (s t : State α) (h : t.mem=s.mem) :
    (preWrite P p s t).mem=(preWrite P p s s).mem := by
  apply scatterMemory
  exact scatterMemory _ _ _ _ _ _ h

theorem preRow_injective (p : PreParameters) (s : State α) : Function.Injective (preRow p s) := by
  intro a b he
  apply Prod.ext
  · apply Fin.ext
    simpa only [preRow,Nat.add_left_cancel_iff] using he
  · exact Subsingleton.elim _ _

theorem preprocessCorrectness (rest : Algebra α) (P : Primitives α) (p : PreParameters) (s : State α)
    (hd : p.newGradient≠p.delta) (hinj : Function.Injective (preOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest P) (preSource p) s=some final ∧
      final.mem=(preWrite P p s s).mem ∧
      (∀ i, final.mem p.newGradient (preOffset p s i)=preGradientCell P p s i) ∧
      (∀ i, final.mem p.delta (preRow p s i)=preDeltaCell P p s i) ∧
      (∀ r o, (r≠p.newGradient ∨ ∀ i, o≠preOffset p s i) →
        (r≠p.delta ∨ ∀ i, o≠preRow p s i) → final.mem r o=s.mem r o) := by
  have hm : (preWrite P p s (preRegisters P p s)).mem=(preWrite P p s s).mem := by
    apply preWrite_memory
    simp [preRegisters]
  refine ⟨_,preprocessExecution rest P p s,hm,?_,?_,?_⟩
  · intro i
    rw [hm]
    unfold preWrite
    rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hd)]
    exact State.scatter_readback _ _ _ _ i hinj
  · intro i
    rw [hm]
    exact State.scatter_readback _ _ _ _ i (preRow_injective p s)
  · intro r o hg hd
    rw [hm]
    unfold preWrite
    rw [State.scatter_frame _ _ _ _ _ _ (hd.imp id (fun h i _ => h i))]
    exact State.scatter_frame _ _ _ _ _ _ (hg.imp id (fun h i _ => h i)) _

#specNonCircular preWrite avoiding [preSource,FP.Structural.exec,run,step]
#axiomsClean preprocessExecution
#axiomsClean preprocessCorrectness
end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
