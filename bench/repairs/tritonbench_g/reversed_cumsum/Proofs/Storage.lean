import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.NativeIndex
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsum.Storage
open VeriTile.Triton Geometry
noncomputable section
abbrev Placement (h : Call) := ReversedCumsumScalar.Storage.Placement h.flat
abbrev address := ReversedCumsumScalar.Storage.address
abbrev decode := ReversedCumsumScalar.Storage.decode
abbrev lift {β : Type} (h : Call) := @ReversedCumsumScalar.Storage.lift β h.flat
abbrev commit {β : Type} (h : Call) := @ReversedCumsumScalar.Storage.commit β h.flat

def writes (h : Call) (p : Placement h) (programs : List (Program h)) (addr : Nat) : Prop :=
  ∃ program∈programs, ∃ t, ∃ j : Fin BS, t<h.T ∧ program.2.val*BS+j.val<h.S ∧
    addr=address p.backing.output (Geometry.address h program.1.val t (program.2.val*BS+j.val))

def programs (h : Call) : List (Program h) := (List.finRange (h.B*h.H)).product (List.finRange (featureChunks h))

theorem window (h : Call) (p : Placement h) (row : Row h) (t s : Nat) (ht : t<h.T) (hs : s<h.S) :
    Geometry.address h row.val t s<p.backing.input.extent ∧ Geometry.address h row.val t s<p.backing.output.extent := by
  have a := p.backing.inputWindow row.val (t*h.S+s) row.isLt (inner_bound h t s ht hs)
  have b := p.backing.outputWindow row.val (t*h.S+s) row.isLt (inner_bound h t s ht hs)
  rw [(flat_address h _ _).1] at a
  rw [(flat_address h _ _).2] at b
  simpa [Geometry.address,Nat.add_assoc] using And.intro a b

theorem program_unique (h : Call) (a b : Program h) (t u : Nat) (i j : Fin BS)
    (ht : t<h.T) (hu : u<h.T) (hi : a.2.val*BS+i.val<h.S) (hj : b.2.val*BS+j.val<h.S)
    (eq : Geometry.address h a.1.val t (a.2.val*BS+i.val)=Geometry.address h b.1.val u (b.2.val*BS+j.val)) : a=b := by
  obtain ⟨rows,times,features⟩ := Geometry.unique h a.1 b.1 t u _ _ ht hu hi hj eq
  have blocks := (Geometry.pair_unique BS a.2.val b.2.val i.val j.val i.isLt j.isLt features).1
  exact Prod.ext rows (Fin.ext blocks)

theorem feature_decompose (h : Call) (s : Nat) (hs : s<h.S) :
    ∃ f : FeatureBlock h, ∃ j : Fin BS, f.val*BS+j.val=s := by
  have cover := ReversedCumsumScalar.Geometry.chunks_cover h.featureView ⟨BS,1⟩ (by decide)
  have div : s/BS<featureChunks h := (Nat.div_lt_iff_lt_mul (by decide : 0<BS)).mpr (hs.trans_le cover)
  refine ⟨⟨s/BS,div⟩,⟨s%BS,Nat.mod_lt _ (by decide)⟩,?_⟩
  simpa [Nat.mul_comm] using Nat.div_add_mod s BS

theorem writes_full (h : Call) (p : Placement h) (addr : Nat) :
    writes h p (programs h) addr ↔
      ReversedCumsumScalar.Storage.writes h.flat p (List.finRange (h.B*h.H)) addr := by
  constructor
  · rintro ⟨program,_,t,j,ht,hs,eq⟩
    refine ⟨program.1,List.mem_finRange program.1,t*h.S+(program.2.val*BS+j.val),Geometry.inner_bound h _ _ ht hs,?_⟩
    rw [(Geometry.flat_address h _ _).2]
    simpa [Geometry.address,Nat.add_assoc] using eq
  · rintro ⟨row,_,u,hu,eq⟩
    change u<h.T*h.S at hu
    have space : 0<h.S := by nlinarith
    have time : u/h.S<h.T := (Nat.div_lt_iff_lt_mul space).mpr hu
    obtain ⟨f,j,hj⟩ := feature_decompose h (u%h.S) (Nat.mod_lt _ space)
    refine ⟨(row,f),by simp [programs],u/h.S,j,time,by rw [hj]; exact Nat.mod_lt _ space,?_⟩
    rw [hj,(Geometry.flat_address h _ _).2] at *
    have reconstruct : u/h.S*h.S+u%h.S=u := by simpa [Nat.mul_comm] using Nat.div_add_mod u h.S
    simpa [Geometry.address,Nat.add_assoc,reconstruct] using eq

#print axioms writes_full
end
end VeriTile.Bench.Repairs.ReversedCumsum.Storage
