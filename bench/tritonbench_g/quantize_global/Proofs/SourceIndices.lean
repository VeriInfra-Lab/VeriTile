import bench.tritonbench_g.quantize_global.Kernels

namespace QuantizeGlobalFinite
theorem footprint_membership (c : Config) (pids : List Nat) (i : Nat) :
    i∈footprint c pids ↔ i/c.block∈pids := by
  simp only [footprint,List.mem_flatMap,List.mem_map,List.mem_range]
  constructor
  · rintro ⟨pid,hp,lane,hl,hi⟩
    have he : i/c.block=pid := by cases c <;> simp [offset,Config.block] at hi hl ⊢ <;> omega
    simpa [he] using hp
  · intro hp
    refine ⟨i/c.block,hp,i%c.block,Nat.mod_lt _ (by cases c <;> decide),?_⟩
    cases c <;> simp [offset,Config.block] <;> omega

end QuantizeGlobalFinite
