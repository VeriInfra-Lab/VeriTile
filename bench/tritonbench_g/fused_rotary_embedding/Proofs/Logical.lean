import bench.tritonbench_g.fused_rotary_embedding.Proofs.Source

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

def logicalOutput (c : Config) (kind : Kind) (t h d : Nat) : Nat :=
  if kind=.q then c.shape.data .q t h d
  else c.shape.cache (kind=.k) (c.block t) h (c.slot t) d
def Footprint (c : Config) (r : Region) (a : Nat) : Prop :=
  ∃ kind t h d,t<c.shape.tokens ∧ h<c.shape.heads kind ∧ d<c.shape.width ∧
    r=kind.output ∧ a=logicalOutput c kind t h d

def columnWork (s : Shape) (kind : Kind) (t h d : Nat) : Work :=
  ⟨kind,t,h,if d<s.half then d else d-s.half⟩
def secondHalf (s : Shape) (d : Nat) : Bool := decide (s.half≤d)

theorem column_work (s : Shape) (hs : s.Valid) (kind : Kind) (t h d : Nat)
    (ht : t<s.tokens) (hh : h<s.heads kind) (hd : d<s.width) :
    s.Active (columnWork s kind t h d) ∧ s.dim (columnWork s kind t h d) (secondHalf s d)=d := by
  have wh := width_halves s hs
  by_cases low : d<s.half
  · simp [columnWork,secondHalf,Shape.Active,Shape.dim,low,ht,hh,show ¬s.half≤d by omega]
  · have sub : d-s.half<s.half := by omega
    simp [columnWork,secondHalf,Shape.Active,Shape.dim,low,ht,hh,sub,show s.half≤d by omega]

theorem output_logical (c : Config) (w : Work) (o : Bool) :
    c.output w o=logicalOutput c w.kind w.token w.head (c.shape.dim w o) := by
  cases kw : w.kind <;> simp [Config.output,logicalOutput,Shape.input,kw]

theorem writes_footprint (c : Config) (hc : c.Valid) (w : Work) (hw : c.shape.Active w)
    (r : Region) (a : Nat) (h : Writes c w r a) : Footprint c r a := by
  rcases h with ⟨rfl,ha|ha⟩
  · exact ⟨w.kind,w.token,w.head,c.shape.dim w false,hw.1,hw.2.1,
      dim_bound c.shape hc.shape w hw false,rfl,ha.trans (output_logical c w false)⟩
  · exact ⟨w.kind,w.token,w.head,c.shape.dim w true,hw.1,hw.2.1,
      dim_bound c.shape hc.shape w hw true,rfl,ha.trans (output_logical c w true)⟩

end VeriTile.Bench.Repairs.FusedRotaryEmbedding
