import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Source

namespace VeriTile.Bench.Repairs.ReversedCumsum.Geometry
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2400000

abbrev chunks (h : Call) (c : Config) := ReversedCumsumScalar.Geometry.chunks h.timeView c
abbrev featureChunks (h : Call) := ReversedCumsumScalar.Geometry.chunks h.featureView ⟨BS,1⟩
abbrev Row (h : Call) := Fin (h.B*h.H)
abbrev FeatureBlock (h : Call) := Fin (featureChunks h)
abbrev Program (h : Call) := Row h × FeatureBlock h

abbrev activeT (h : Call) (c : Config) (q : Nat) (i : Fin c.BT) := i.val<h.T-q*c.BT
abbrev activeS (h : Call) (f : Nat) (j : Fin BS) := j.val<h.S-f*BS
abbrev active (h : Call) (c : Config) (q f : Nat) (i : Fin c.BT) (j : Fin BS) := activeT h c q i ∧ activeS h f j

def time (h : Call) (c : Config) (q : Nat) (i : Fin c.BT) : Nat := q*c.BT+if activeT h c q i then i.val else 0
def feature (h : Call) (f : Nat) (j : Fin BS) : Nat := f*BS+if activeS h f j then j.val else 0

def address (h : Call) (row t s : Nat) : Nat := row*(h.T*h.S)+t*h.S+s

theorem chunk_start (h : Call) (c : Config) (positive : 0<c.BT) (q : Nat) (hq : q<chunks h c) : q*c.BT<h.T :=
  ReversedCumsumScalar.Geometry.chunk_start h.timeView c positive q hq

theorem feature_start (h : Call) (f : FeatureBlock h) : f.val*BS<h.S :=
  ReversedCumsumScalar.Geometry.chunk_start h.featureView ⟨BS,1⟩ (by decide) f.val f.isLt

theorem time_bound (h : Call) (c : Config) (positive : 0<c.BT) (q : Nat) (hq : q<chunks h c) (i : Fin c.BT) :
    time h c q i<h.T := ReversedCumsumScalar.Geometry.time_bound h.timeView c positive q hq i

theorem feature_bound (h : Call) (f : FeatureBlock h) (j : Fin BS) : feature h f.val j<h.S :=
  ReversedCumsumScalar.Geometry.time_bound h.featureView ⟨BS,1⟩ (by decide) f.val f.isLt j

theorem active_iff (h : Call) (c : Config) (q : Nat) (f : FeatureBlock h) (i : Fin c.BT) (j : Fin BS)
    (start : q*c.BT<h.T) : active h c q f.val i j ↔ q*c.BT+i.val<h.T ∧ f.val*BS+j.val<h.S := by
  have fs := feature_start h f
  simp only [active,activeT,activeS]
  omega

theorem flat_address (h : Call) (row u : Nat) :
    h.flat.inputStrides.address h.flat.H row u=row*(h.T*h.S)+u ∧
    h.flat.outputStrides.address h.flat.H row u=row*(h.T*h.S)+u := by
  have calcEq : row/h.H*(h.H*(h.T*h.S))+row%h.H*(h.T*h.S)+u=row*(h.T*h.S)+u := by
    calc
      _ = (h.H*(row/h.H)+row%h.H)*(h.T*h.S)+u := by ring
      _ = _ := by rw [Nat.div_add_mod]
  simpa [Call.flat,ReversedCumsumScalar.Strides.address,ReversedCumsumScalar.Strides.base] using And.intro calcEq calcEq

theorem inner_bound (h : Call) (t s : Nat) (ht : t<h.T) (hs : s<h.S) : t*h.S+s<h.T*h.S := by nlinarith

theorem pair_unique (size a b x y : Nat) (hx : x<size) (hy : y<size)
    (eq : a*size+x=b*size+y) : a=b ∧ x=y := by
  have mods := congrArg (fun n => n%size) eq
  simp only [Nat.add_mod,Nat.mul_mod_left,Nat.zero_add,Nat.mod_eq_of_lt hx,Nat.mod_eq_of_lt hy] at mods
  refine ⟨?_,mods⟩
  have pos := Nat.zero_lt_of_lt hx
  nlinarith

theorem unique (h : Call) (a b : Row h) (t u x y : Nat)
    (ht : t<h.T) (hu : u<h.T) (hx : x<h.S) (hy : y<h.S)
    (eq : address h a.val t x=address h b.val u y) : a=b ∧ t=u ∧ x=y := by
  have outer := pair_unique (h.T*h.S) a.val b.val (t*h.S+x) (u*h.S+y)
    (inner_bound h t x ht hx) (inner_bound h u y hu hy) (by simpa [address,Nat.add_assoc] using eq)
  exact ⟨Fin.ext outer.1,pair_unique h.S t u x y hx hy outer.2⟩

theorem flat_layout (h : Call) (distinct : h.input≠h.output) : ReversedCumsumScalar.Geometry.OutputLayout h.flat where
  distinct := distinct
  unique := by
    intro a b x y ha hb hx hy eq
    rw [(flat_address h a x).2,(flat_address h b y).2] at eq
    exact pair_unique (h.T*h.S) a b x y hx hy eq

#print axioms flat_layout
end
end VeriTile.Bench.Repairs.ReversedCumsum.Geometry
