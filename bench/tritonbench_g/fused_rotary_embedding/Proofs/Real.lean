import bench.tritonbench_g.fused_rotary_embedding.Proofs.Logical

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

/-- Independent column-indexed split-half rotation of initial inputs. The
coefficient row is the token row supplied by the caller. V is a pure copy. -/
noncomputable def formula (s : Shape) (initial : Memory ℝ) (kind : Kind) (t h d : Nat) : ℝ :=
  if kind=.v then initial .v (s.data .v t h d)
  else if d<s.half then
    initial kind.input (s.data kind t h d)*initial .cos (s.coeff false t d) -
    initial kind.input (s.data kind t h (d+s.half))*initial .sin (s.coeff true t d)
  else initial kind.input (s.data kind t h (d-s.half))*initial .sin (s.coeff true t (d-s.half)) +
    initial kind.input (s.data kind t h d)*initial .cos (s.coeff false t (d-s.half))

theorem pair_real_formula (s : Shape) (hs : s.Valid) (w : Work) (hw : s.Active w)
    (m : Memory ℝ) (o : Bool) :
    component (pairValues s (fun _ => realOps) w m) o=formula s m w.kind w.token w.head (s.dim w o) := by
  have bound := hw.2.2
  have high : ¬w.pair+s.half<s.half := by omega
  cases kw : w.kind <;> cases o <;>
    simp [component,pairValues,sourcePair,realOps,formula,Shape.dim,Shape.input,kw,Kind.input,bound,high]

theorem actual_formula (c : Config) (hc : c.Valid) (m : Memory ℝ) (kind : Kind) (t h d : Nat)
    (ht : t<c.shape.tokens) (hh : h<c.shape.heads kind) (hd : d<c.shape.width) :
    actualSource c (fun _ => realOps) m kind.output (logicalOutput c kind t h d)=formula c.shape m kind t h d := by
  let w := columnWork c.shape kind t h d
  let o := secondHalf c.shape d
  have hw := column_work c.shape hc.shape kind t h d ht hh hd
  have result := run_readback c hc (fun _ => realOps) m w hw.1 o
  rw [pair_real_formula c.shape hc.shape w hw.1 m o,output_logical] at result
  have dimEq : c.shape.dim w o=d := hw.2
  rw [dimEq] at result
  rw [actualSource_eq_run c hc]
  simpa [w,columnWork] using result

def RealPost (c : Config) (initial final : Memory ℝ) : Prop :=
  (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
    final kind.output (logicalOutput c kind t h d)=formula c.shape initial kind t h d) ∧
  (∀ r a,¬Footprint c r a → final r a=initial r a)

theorem real_post (c : Config) (hc : c.Valid) (m : Memory ℝ) :
    RealPost c m (actualSource c (fun _ => realOps) m) := by
  constructor
  · exact actual_formula c hc m
  · intro r a outside
    rw [actualSource_eq_run c hc]
    apply run_frame
    intro w hw h
    exact outside (writes_footprint c hc w hw r a h)

#specNonCircular formula avoiding [sourcePair,pairValues,actualSource,run]
#axiomsClean actual_formula
#axiomsClean real_post
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
