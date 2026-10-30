import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.IO
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.RawGrid
import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Bounds

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem binding_metadata [Inhabited β] (rest : Algebra β) (ops : Ops β)
    (encode : α → β) (observe : Kind → β → α) (c : Config) (hc : c.Valid)
    (xs : ChannelValues α (inputChannels c.shape)) (domain : metadataDomain c xs) (s : State β)
    (binding : (io rest ops encode observe c).InputBinding xs s) : Source.MetadataBinding c s.mem := by
  constructor
  · intro t ht
    exact (binding ⟨5,by simp [io]⟩ (⟨t,ht⟩,PUnit.unit) trivial).trans (domain.1 ⟨t,ht⟩)
  · intro t ht
    exact (binding ⟨6,by simp [io]⟩ (⟨_,table_bound c hc t ht⟩,PUnit.unit) trivial).trans
      (domain.2 t ht _)

theorem binding_data [Inhabited α] [Inhabited β] (rest : Algebra β) (ops : Ops β)
    (encode : α → β) (observe : Kind → β → α) (c : Config)
    (xs : ChannelValues α (inputChannels c.shape)) (s : State β)
    (binding : (io rest ops encode observe c).InputBinding xs s)
    (kind : Kind) (a : Nat) (ha : a<c.shape.tokens*c.shape.heads kind*c.shape.width) :
    Source.rawValues s.mem kind.input a = encode (inputMemory c.shape xs kind.input a) := by
  cases kind
  all_goals simp [Shape.heads] at ha
  · have got := binding ⟨0,by simp [io]⟩ (⟨a,ha⟩,PUnit.unit) trivial
    change (s.mem "q" a).read .real = encode (xs ⟨0,by simp⟩ (⟨a,ha⟩,PUnit.unit)) at got
    simpa [Source.rawValues,Source.region,Kind.input,inputMemory,vectorAt,ha] using got
  · have got := binding ⟨1,by simp [io]⟩ (⟨a,ha⟩,PUnit.unit) trivial
    change (s.mem "k" a).read .real = encode (xs ⟨1,by simp⟩ (⟨a,ha⟩,PUnit.unit)) at got
    simpa [Source.rawValues,Source.region,Kind.input,inputMemory,vectorAt,ha] using got
  · have got := binding ⟨2,by simp [io]⟩ (⟨a,ha⟩,PUnit.unit) trivial
    change (s.mem "v" a).read .real = encode (xs ⟨2,by simp⟩ (⟨a,ha⟩,PUnit.unit)) at got
    simpa [Source.rawValues,Source.region,Kind.input,inputMemory,vectorAt,ha] using got

theorem binding_coefficient [Inhabited α] [Inhabited β] (rest : Algebra β) (ops : Ops β)
    (encode : α → β) (observe : Kind → β → α) (c : Config)
    (xs : ChannelValues α (inputChannels c.shape)) (s : State β)
    (binding : (io rest ops encode observe c).InputBinding xs s)
    (sine : Bool) (a : Nat) (ha : a<c.shape.tokens*(if sine then c.shape.sinWidth else c.shape.cosWidth)) :
    Source.rawValues s.mem (if sine then .sin else .cos) a =
      encode (inputMemory c.shape xs (if sine then .sin else .cos) a) := by
  cases sine
  all_goals simp only [Bool.false_eq_true,ite_true,ite_false] at ha
  · have got := binding ⟨3,by simp [io]⟩ (⟨a,ha⟩,PUnit.unit) trivial
    change (s.mem "cos" a).read .real = encode (xs ⟨3,by simp⟩ (⟨a,ha⟩,PUnit.unit)) at got
    simpa [Source.rawValues,Source.region,inputMemory,vectorAt,ha] using got
  · have got := binding ⟨4,by simp [io]⟩ (⟨a,ha⟩,PUnit.unit) trivial
    change (s.mem "sin" a).read .real = encode (xs ⟨4,by simp⟩ (⟨a,ha⟩,PUnit.unit)) at got
    simpa [Source.rawValues,Source.region,inputMemory,vectorAt,ha] using got

theorem outside_footprint (rest : Algebra β) (ops : Ops β) (encode : α → β)
    (observe : Kind → β → α) (c : Config) (xs : ChannelValues α (inputChannels c.shape))
    (r : RegionName) (a : Nat) (outside : (io rest ops encode observe c).Outside xs r a) :
    ¬Footprint c (Source.decoded r) a := by
  rintro ⟨kind,t,h,d,ht,hh,hd,hr,ha⟩
  have inverse : r=Source.region kind.output := by
    cases kind <;> simp only [Source.decoded,Kind.output] at hr <;>
      split_ifs at hr <;> simp_all [Source.region,Kind.output]
  cases kind
  · exact (outside.1 ⟨0,by simp [io]⟩ (⟨t,ht⟩,⟨h,hh⟩,⟨d,hd⟩,PUnit.unit) trivial).elim
      (fun no=>no inverse) (fun no=>no ha)
  · exact (outside.1 ⟨1,by simp [io]⟩ (⟨t,ht⟩,⟨h,hh⟩,⟨d,hd⟩,PUnit.unit) trivial).elim
      (fun no=>no inverse) (fun no=>no ha)
  · exact (outside.1 ⟨2,by simp [io]⟩ (⟨t,ht⟩,⟨h,hh⟩,⟨d,hd⟩,PUnit.unit) trivial).elim
      (fun no=>no inverse) (fun no=>no ha)

#axiomsClean binding_metadata
#axiomsClean binding_data
#axiomsClean binding_coefficient
#axiomsClean outside_footprint
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
