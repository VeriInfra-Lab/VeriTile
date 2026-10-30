import bench.tritonbench_g.fused_rotary_embedding.Proofs.IOLayout
import bench.tritonbench_g.fused_rotary_embedding.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem bounds_data (rest : Algebra β) (ops : Ops β) (encode : α → β)
    (observe : Kind → β → α) (c : Config) (xs : ChannelValues α (inputChannels c.shape))
    (bounds : RegionName → Nat) (hb : (io rest ops encode observe c).InBounds bounds xs)
    (kind : Kind) (t h d : Nat) (ht : t<c.shape.tokens) (hh : h<c.shape.heads kind) (hd : d<c.shape.width) :
    c.shape.data kind t h d < bounds (Source.region kind.input) := by
  have ha : c.shape.data kind t h d < c.shape.tokens*c.shape.heads kind*c.shape.width :=
    radix_bound _ _ _ _ (radix_bound _ _ _ _ ht hh) hd
  cases kind
  · exact hb.2.1 ⟨0,by simp [io]⟩ (⟨_,ha⟩,PUnit.unit) trivial
  · exact hb.2.1 ⟨1,by simp [io]⟩ (⟨_,ha⟩,PUnit.unit) trivial
  · exact hb.2.1 ⟨2,by simp [io]⟩ (⟨_,ha⟩,PUnit.unit) trivial

theorem bounds_coefficient (rest : Algebra β) (ops : Ops β) (encode : α → β)
    (observe : Kind → β → α) (c : Config) (hc : c.Valid) (xs : ChannelValues α (inputChannels c.shape))
    (bounds : RegionName → Nat) (hb : (io rest ops encode observe c).InBounds bounds xs)
    (sine : Bool) (t d : Nat) (ht : t<c.shape.tokens) (hd : d<c.shape.half) :
    c.shape.coeff sine t d < bounds (if sine then "sin" else "cos") := by
  have ha : c.shape.coeff sine t d<c.shape.tokens*(if sine then c.shape.sinWidth else c.shape.cosWidth) := by
    apply radix_bound _ _ _ _ ht
    cases sine
    · exact lt_of_lt_of_le hd hc.shape.cos
    · exact lt_of_lt_of_le hd hc.shape.sin
  cases sine
  · exact hb.2.1 ⟨3,by simp [io]⟩ (⟨_,ha⟩,PUnit.unit) trivial
  · exact hb.2.1 ⟨4,by simp [io]⟩ (⟨_,ha⟩,PUnit.unit) trivial

theorem bounds_output (rest : Algebra β) (ops : Ops β) (encode : α → β)
    (observe : Kind → β → α) (c : Config) (xs : ChannelValues α (inputChannels c.shape))
    (bounds : RegionName → Nat) (hb : (io rest ops encode observe c).InBounds bounds xs)
    (kind : Kind) (t h d : Nat) (ht : t<c.shape.tokens) (hh : h<c.shape.heads kind) (hd : d<c.shape.width) :
    logicalOutput c kind t h d < bounds (Source.region kind.output) := by
  cases kind
  · exact hb.2.2.1 ⟨0,by simp [io]⟩ (⟨t,ht⟩,⟨h,hh⟩,⟨d,hd⟩,PUnit.unit) trivial
  · exact hb.2.2.1 ⟨1,by simp [io]⟩ (⟨t,ht⟩,⟨h,hh⟩,⟨d,hd⟩,PUnit.unit) trivial
  · exact hb.2.2.1 ⟨2,by simp [io]⟩ (⟨t,ht⟩,⟨h,hh⟩,⟨d,hd⟩,PUnit.unit) trivial

theorem program_bounds [Inhabited β] (rest : Algebra β) (ops : Ops β) (encode : α → β)
    (observe : Kind → β → α) (c : Config) (hc : c.Valid) (xs : ChannelValues α (inputChannels c.shape))
    (bounds : RegionName → Nat) (hb : (io rest ops encode observe c).InBounds bounds xs)
    (pid : Program) (hp : pid ∈ programs c.shape) (m : Source.Mem β)
    (hm : Source.MetadataBinding c m) :
    Source.AccessBounds (Source.parameters c.shape) (Source.state pid m) bounds := by
  have ht := List.mem_range.mp (List.mem_product.mp hp).2
  have activeWork (key : Bool) (i : Source.HalfLane (Source.parameters c.shape))
      (hi : Source.active (Source.parameters c.shape) i) : c.shape.Active (Source.work c.shape pid key i.1.val) := by
    have hh := program_head_bound c.shape hc.shape pid hp (if key then .k else .q)
    have hd := (Source.active_nat _ _).mp hi
    cases key <;> simpa [Source.work,Shape.Active,programHead,ht,hd] using hh
  constructor
  · intro high i hi
    rw [Source.dataOffset_nat]
    have hw := activeWork Bool.false i hi
    exact bounds_data rest ops encode observe c xs bounds hb .q pid.2 pid.1 _ ht hw.2.1
      (dim_bound c.shape hc.shape _ hw high)
  · intro sine i hi
    rw [Source.coefficientOffset_nat]
    exact bounds_coefficient rest ops encode observe c hc xs bounds hb sine pid.2 i.1.val ht
      ((Source.active_nat _ _).mp hi)
  · intro _ high i hi
    rw [Source.dataOffset_nat]
    have hw := activeWork Bool.true i hi
    exact bounds_data rest ops encode observe c xs bounds hb .k pid.2 (pid.1/c.shape.groups) _ ht hw.2.1
      (dim_bound c.shape hc.shape _ hw high)
  · intro _
    simpa [Source.state,Source.parameters] using hb.2.1 ⟨5,by simp [io]⟩ (⟨pid.2,ht⟩,PUnit.unit) trivial
  · intro _
    rw [Source.raw_past c hc pid ht m hm]
    simp only [Source.state,Source.parameters,Source.quot_nat,Nat.one_ne_zero,ite_false,ite_true,
      Nat.cast_one,Int.mul_one,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    exact hb.2.1 ⟨6,by simp [io]⟩ (⟨_,table_bound c hc pid.2 ht⟩,PUnit.unit) trivial
  · intro _ high i hi
    rw [Source.raw_cacheOffset c hc pid ht m hm high i ((Source.active_nat _ _).mp hi),output_logical]
    have hw := activeWork Bool.true i hi
    exact bounds_output rest ops encode observe c xs bounds hb .k pid.2 (pid.1/c.shape.groups) _ ht hw.2.1
      (dim_bound c.shape hc.shape _ hw high)
  · intro _ i hi
    rw [Source.valueOffset_nat]
    exact bounds_data rest ops encode observe c xs bounds hb .v pid.2 (pid.1/c.shape.groups) _ ht
      (program_head_bound c.shape hc.shape pid hp .v) hi
  · intro _ i hi
    rw [Source.raw_valueCacheOffset c hc pid ht m hm]
    exact bounds_output rest ops encode observe c xs bounds hb .v pid.2 (pid.1/c.shape.groups) _ ht
      (program_head_bound c.shape hc.shape pid hp .v) hi

theorem bounds_mem_congr [Inhabited β] (p : Source.Parameters) (s t : State β)
    (hm : s.mem=t.mem) (hp : s.pids=t.pids) (bounds : RegionName → Nat)
    (hb : Source.AccessBounds p t bounds) : Source.AccessBounds p s bounds := by
  rcases hb with ⟨hq,hc,hk,hl,ht,hkc,hv,hvc⟩
  constructor <;>
    simp_all only [Source.dataOffset,Source.coefficientOffset,Source.cacheOffset,Source.valueOffset,
      Source.valueCacheOffset,Source.past,Source.block,Source.slot,Source.head,hm,hp]
  all_goals simp

theorem schedule_safe [Inhabited β] (rest : Algebra β) (ops : Ops β) (encode : α → β)
    (observe : Kind → β → α) (c : Config) (hc : c.Valid) (xs : ChannelValues α (inputChannels c.shape))
    (bounds : RegionName → Nat) (hb : (io rest ops encode observe c).InBounds bounds xs)
    (ps : List Program) (inside : ∀ pid∈ps,pid∈programs c.shape) (s : State β)
    (hm : Source.MetadataBinding c s.mem) :
    ScheduleSafe bounds (ps.map (Source.invocation rest ops c)) s := by
  induction ps generalizing s with
  | nil => trivial
  | cons pid ps ih =>
    refine ⟨?_,?_⟩
    · apply Source.source_safe
      apply bounds_mem_congr _ _ (Source.state pid s.mem) (by simp [Source.state])
        (by simp [Source.invocation,Invocation.initial,bindRegisters,Source.state]) bounds
      exact program_bounds rest ops encode observe c hc xs bounds hb pid (inside pid (by simp)) s.mem hm
    · intro final executed
      have ht := List.mem_range.mp (List.mem_product.mp (inside pid (by simp))).2
      obtain ⟨next,run,memory⟩ := Source.invocation_memory rest ops c hc pid ht s hm
      rw [run] at executed
      cases Option.some.inj executed
      apply ih (fun p hp=>inside p (by simp [hp]))
      rw [memory]
      exact Source.rawProgram_metadata c ops pid s.mem hm

#axiomsClean program_bounds
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
