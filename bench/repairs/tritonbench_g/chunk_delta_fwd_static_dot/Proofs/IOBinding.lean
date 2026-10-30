import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.IO
import bench.repairs.tritonbench_g.chunk_delta_fwd.GridOutputs

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.IOBinding
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 8000000
variable {α β : Type} [Inhabited α] [Inhabited β]

/-- Only logical tensor coordinates are bound; unused surrounding input
storage remains arbitrary. Coincident weight addresses retain one read value. -/
structure Reads (p : Parameters) (encode : α → β) (xs : Inputs α p) (m : Memory β) : Prop where
  key : ∀ head t k, head < p.B*p.H → t < p.T → k < p.K → 0 < p.V →
    (m "key" (head*(p.T*p.K)+k+t*p.K)).read .real = encode (extend (key xs) head t k)
  weight : ∀ head t k, head < p.B*p.H → t < p.T → k < p.K → 0 < p.V →
    (m "weight" (head/p.H*p.wBatch+head%p.H*p.wHead+t*p.wTime+k*p.wDim)).read .real =
      encode (extend (weight xs) head t k)
  value : ∀ head t v, head < p.B*p.H → t < p.T → v < p.V →
    (m "value" (head*(p.T*p.V)+t*p.V+v)).read .real = encode (extend (value xs) head t v)
  seed : p.useSeed = Bool.true → ∀ head k v, head < p.B*p.H → k < p.K → v < p.V →
    (m "initial" (head*p.K*p.V+k*p.V+v)).read .real = encode (extend (seed xs) head k v)

theorem reads (M : Algebra β) (D : DotPrimitive β) (p : Parameters) (schedule : List PID)
    (encode : α → β) (observe : β → α) (xs : Inputs α p) (s : State β)
    (h : (io M D p schedule encode observe).InputBinding xs s) : Reads p encode xs s.mem := by
  constructor
  · intro head t k hh ht hk hv
    have hb := h (0 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨k,hk⟩,PUnit.unit) hv
    simpa [io,ProgramIO.InputBinding,inputs,Interface.read,key,extend,hh,ht,hk,mapValue] using hb
  · intro head t k hh ht hk hv
    have hb := h (1 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨k,hk⟩,PUnit.unit) hv
    simpa [io,ProgramIO.InputBinding,inputs,Interface.read,weight,extend,hh,ht,hk,mapValue] using hb
  · intro head t v hh ht hv
    have hb := h (2 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨v,hv⟩,PUnit.unit) trivial
    simpa [io,ProgramIO.InputBinding,inputs,Interface.read,value,extend,hh,ht,hv,mapValue] using hb
  · intro on head k v hh hk hv
    have hb := h (3 : Fin 4) (⟨head,hh⟩,⟨k,hk⟩,⟨v,hv⟩,PUnit.unit) on
    simpa [io,ProgramIO.InputBinding,inputs,Interface.read,seed,extend,hh,hk,hv,mapValue] using hb

theorem packed (n m o : Nat) (ho : o < n*m) :
    ∃ a : Fin n, ∃ b : Fin m, o = a.val*m+b.val := by
  have hm : 0 < m := by
    by_contra h
    have hz : m=0 := by omega
    simp [hz] at ho
  refine ⟨⟨o/m,(Nat.div_lt_iff_lt_mul hm).mpr ho⟩,⟨o%m,Nat.mod_lt _ hm⟩,?_⟩
  simpa [Nat.mul_comm,Nat.add_comm] using (Nat.mod_add_div o m).symm

theorem outside (M : Algebra β) (D : DotPrimitive β) (p : Parameters) (schedule : List PID)
    (encode : α → β) (observe : β → α) (xs : Inputs α p) (r : RegionName) (o : Nat)
    (h : (io M D p schedule encode observe).Outside xs r o) : ¬OutputCell p r o := by
  intro output
  rcases output with ⟨rfl,ho⟩|⟨rfl,ho⟩|⟨rfl,on,ho⟩
  · obtain ⟨a,v,hav⟩ := packed (p.B*p.H*p.NT*p.K) p.V o ho
    obtain ⟨b,k,hbk⟩ := packed (p.B*p.H*p.NT) p.K a.val a.isLt
    obtain ⟨head,c,hhc⟩ := packed (p.B*p.H) p.NT b.val b.isLt
    have he := h.1 (0 : Fin 3) (head,c,k,v,PUnit.unit) trivial
    exact he.elim (fun hn => hn rfl) (fun hn => hn (by change o = _; rw [hav,hbk,hhc]; rfl))
  · obtain ⟨a,v,hav⟩ := packed (p.B*p.H*p.T) p.V o ho
    obtain ⟨head,t,hht⟩ := packed (p.B*p.H) p.T a.val a.isLt
    have he := h.1 (1 : Fin 3) (head,t,v,PUnit.unit) trivial
    exact he.elim (fun hn => hn rfl) (fun hn => hn (by change o = _; rw [hav,hht]; rfl))
  · obtain ⟨a,v,hav⟩ := packed (p.B*p.H*p.K) p.V o ho
    obtain ⟨head,k,hhk⟩ := packed (p.B*p.H) p.K a.val a.isLt
    have he := h.1 (2 : Fin 3) (head,k,v,PUnit.unit) on
    exact he.elim (fun hn => hn rfl) (fun hn => hn (by change o = _; rw [hav,hhk]; rfl))

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.IOBinding
