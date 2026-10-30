import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelContract
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelRealLoop

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelReal
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem split_count (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) : (config m b).dims/(config m b).width=(config m b).splits := by
  have hg := split_geometry m b l t h
  rw [←hg.2]
  exact Nat.mul_div_cancel _ hg.1

theorem position_eq (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (s : BlockState) (binding : KernelMetadataBinding m b l t s)
    (hp : ValidProgram (config m b) s.pids) : position s=(past l (s.pids 0):Int) := by
  unfold position
  rw [binding.lengths _ hp.1]
  have bounds := (h.bounds _ hp.1).1
  unfold past
  omega

theorem selectedBlock_eq (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (s : BlockState) (binding : KernelMetadataBinding m b l t s)
    (hp : ValidProgram (config m b) s.pids) :
    selectedBlock (parameters (config m b)) s=(blockId (config m b) l t (s.pids 0):Int) := by
  unfold selectedBlock
  rw [position_eq m b l t h s binding hp]
  simp only [parameters,Int.natCast_tdiv_eq_ediv,←Int.natCast_ediv,Nat.cast_one,Int.mul_one,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero]
  have bounds := metadata_bounds m b l t h _ hp.1
  change s.readMemValue .int "table" (s.pids 0*(config m b).columns+column (config m b) l (s.pids 0))=_
  rw [binding.table _ _ hp.1 bounds.2.1]
  exact (Int.toNat_of_nonneg (h.bounds _ hp.1).2.2.1).symm

theorem sourceOffset_eq (p : Config) (ids : Nat → Nat) (key : Bool) (q : Nat)
    (i : TileIndex [p.width]) :
    sourceOffset (parameters p) ids key q i=inputIndex p (ids 0) (ids 1) (q*p.width+i.1.val) := by
  cases key <;> simp [sourceOffset,parameters,inputIndex,Nat.add_mul,Nat.mul_assoc]

theorem targetInt_eq (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (s : BlockState) (binding : KernelMetadataBinding m b l t s)
    (hp : ValidProgram (config m b) s.pids) (key : Bool) (q : Nat) (hq : q<(config m b).splits)
    (i : TileIndex [(config m b).width]) :
    (targetInt (parameters (config m b)) s key q i).toNat=
      outputIndex (config m b) l t key (s.pids 0) (s.pids 1) (q*(config m b).width+i.1.val) := by
  have expanded := output_expanded m b l t h ⟨⟨s.pids 0,s.pids 1,q⟩,key,i.1.val⟩ ⟨⟨hp.1,hp.2,hq⟩,i.1.isLt⟩
  unfold targetInt
  rw [selectedBlock_eq m b l t h s binding hp,position_eq m b l t h s binding hp]
  cases key <;> simp only [parameters,ite_true,Bool.false_eq_true,ite_false,
    ←Int.ofNat_tmod,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero,Nat.mul_one] at *
  all_goals exact expanded

theorem target_unique (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (s : BlockState) (binding : KernelMetadataBinding m b l t s)
    (hp : ValidProgram (config m b) s.pids) (key : Bool)
    (u v : Nat) (hu : u<(config m b).splits) (hv : v<(config m b).splits)
    (i j : TileIndex [(config m b).width])
    (same : (targetInt (parameters (config m b)) s key u i).toNat=
      (targetInt (parameters (config m b)) s key v j).toNat) : u=v ∧ i=j := by
  rw [targetInt_eq m b l t h s binding hp key u hu i,
    targetInt_eq m b l t h s binding hp key v hv j] at same
  have eq := KvCacheCopy.target_injective m b l t h
    ⟨⟨s.pids 0,s.pids 1,u⟩,key,i.1.val⟩ ⟨⟨s.pids 0,s.pids 1,v⟩,key,j.1.val⟩
    ⟨⟨hp.1,hp.2,hu⟩,i.1.isLt⟩ ⟨⟨hp.1,hp.2,hv⟩,j.1.isLt⟩
    (by simp only [targetAddress,address,component];rw [same])
  exact ⟨congrArg (fun e : Lane => e.tile.split) eq,
    Prod.ext (Fin.ext (congrArg Lane.lane eq)) (Subsingleton.elim _ _)⟩

theorem correctness (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (s : BlockState) (binding : KernelMetadataBinding m b l t s)
    (hp : ValidProgram (config m b) s.pids) :
    ∃ final,exec (Source.originalKernel (parameters (config m b))).toAlgKernel s=some final ∧
      KernelCopy (config m b) l t s.pids s.readMem final.readMem ∧
      (∀ (r : RegionName) o,(∀ key,r≠KvCacheCopy.outputRegion key ∨
        ∀ d,d<(config m b).dims → o≠outputIndex (config m b) l t key (s.pids 0) (s.pids 1) d) →
        final.mem r o=s.mem r o) := by
  obtain ⟨final,he,hm⟩ := execute (parameters (config m b)) s
  have splitEq := split_count m b l t h
  have geometry := split_geometry m b l t h
  refine ⟨final,he,?_,?_⟩
  · intro key d hd
    let i : TileIndex [(config m b).width] := (⟨d%(config m b).width,Nat.mod_lt _ geometry.1⟩,PUnit.unit)
    have hq : d/(config m b).width<(config m b).splits :=
      (Nat.div_lt_iff_lt_mul geometry.1).mpr (by simpa [geometry.2] using hd)
    have hv := copied_readback (parameters (config m b)) s (config m b).splits key _ hq i
      (target_unique m b l t h s binding hp key)
    rw [targetInt_eq m b l t h s binding hp key _ hq i,sourceOffset_eq] at hv
    have offset : d/(config m b).width*(config m b).width+i.1.val=d := by simpa [i,Nat.mul_comm] using Nat.div_add_mod d (config m b).width
    rw [offset] at hv
    change final.readMem (outputRegion key) _=s.readMem (inputRegion key) _
    simpa only [BlockState.readMem,hm,parameters,splitEq] using hv
  · intro r o outside
    rw [hm]
    apply copied_frame
    intro key q hq
    change q<(config m b).dims/(config m b).width at hq
    rw [splitEq] at hq
    rcases outside key with other | miss
    · exact Or.inl other
    · right
      intro i
      rw [targetInt_eq m b l t h s binding hp key q hq i]
      apply miss
      rw [←geometry.2]
      exact Linear.linear_bound _ _ _ _ hq i.1.isLt

#axiomsClean correctness
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelReal
