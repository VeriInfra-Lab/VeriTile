import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelContract
import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelReal
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.KernelReal
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem position_eq (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : BlockState)
    (binding : KernelMetadataBinding m n b l t s)
    (hp : ValidTile (config m n b) (programTile s.pids)) :
    position (parameters (config m n b)) s = (past (config m n b) l (s.pids 0):Int) := by
  simp only [position,sequence,parameters,Int.natCast_tdiv_eq_ediv,←Int.natCast_ediv,
    Int.toNat_natCast,Region.cast]
  change s.readMemValue .int "lengths" (s.pids 0/n) +
    ((s.pids 0:Int)-(n:Int)*((s.pids 0/n:Nat)+1)) = _
  rw [binding.lengths (s.pids 0/n) (sequence_bound m n b l t h (s.pids 0) hp.1)]
  exact position_arithmetic m n b l t h _ hp.1

theorem selectedBlock_eq (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : BlockState)
    (binding : KernelMetadataBinding m n b l t s)
    (hp : ValidTile (config m n b) (programTile s.pids)) :
    selectedBlock (parameters (config m n b)) s = (blockId (config m n b) l t (s.pids 0):Int) := by
  unfold selectedBlock
  rw [position_eq m n b l t h s binding hp]
  simp only [sequence,parameters,Int.natCast_tdiv_eq_ediv,←Int.natCast_ediv,
    ←Int.natCast_mul,Int.mul_one,Int.toNat_natCast,Nat.mul_one,Region.cast]
  have bounds := metadata_bounds m n b l t h _ hp.1
  change s.readMemValue .int "table" (s.pids 0/n*(config m n b).columns +
    column (config m n b) l (s.pids 0)) = _
  rw [binding.table (s.pids 0/n) (column (config m n b) l (s.pids 0))
    (sequence_bound m n b l t h (s.pids 0) hp.1) bounds.2.1]
  unfold blockId
  exact (Int.toNat_of_nonneg (h.bounds.2 _ hp.1).1).symm

theorem sourceOffset_eq (p : Config) (ids : Nat → Nat) (i : TileIndex [p.width]) :
    sourceOffset (parameters p) ids i = inputIndex p (ids 0) (ids 1) (ids 2*p.width+i.1.val) := by
  simp [sourceOffset,parameters,inputIndex,Nat.add_mul,Nat.mul_assoc]

theorem targetOffset_eq (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : BlockState)
    (binding : KernelMetadataBinding m n b l t s)
    (hp : ValidTile (config m n b) (programTile s.pids))
    (i : TileIndex [(config m n b).width]) :
    targetOffset (parameters (config m n b)) s i =
      outputIndex (config m n b) l t (s.pids 0) (s.pids 1) (s.pids 2*(config m n b).width+i.1.val) := by
  unfold targetOffset
  rw [selectedBlock_eq m n b l t h s binding hp,position_eq m n b l t h s binding hp]
  simp only [parameters,←Int.ofNat_tmod,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  exact output_expanded m n b l t h ⟨programTile s.pids,i.1.val⟩ ⟨hp,i.1.isLt⟩

theorem target_injective (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : BlockState)
    (binding : KernelMetadataBinding m n b l t s)
    (hp : ValidTile (config m n b) (programTile s.pids)) :
    Function.Injective (targetOffset (parameters (config m n b)) s) := by
  intro i j same
  rw [targetOffset_eq m n b l t h s binding hp,targetOffset_eq m n b l t h s binding hp] at same
  have eq := KcacheCopyTriton.target_injective m n b l t h
    ⟨programTile s.pids,i.1.val⟩ ⟨programTile s.pids,j.1.val⟩ ⟨hp,i.1.isLt⟩ ⟨hp,j.1.isLt⟩
    (by simp only [targetAddress,address,component,programTile];rw [same])
  have lane := congrArg Lane.lane eq
  exact Prod.ext (Fin.ext lane) (Subsingleton.elim _ _)

theorem correctness (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : BlockState)
    (binding : KernelMetadataBinding m n b l t s)
    (hp : ValidTile (config m n b) (programTile s.pids)) :
    ∃ final,exec (Source.originalKernel (parameters (config m n b))).toAlgKernel s=some final ∧
      KernelCopy (config m n b) l t s.pids (s.readMem "key") (final.readMem "cache") ∧
      (∀ (r : RegionName) o,r≠"cache" ∨ (∀ c,c<(config m n b).width →
        o≠outputIndex (config m n b) l t (s.pids 0) (s.pids 1) (s.pids 2*(config m n b).width+c)) →
        final.mem r o=s.mem r o) := by
  let p := parameters (config m n b)
  refine ⟨(execution p s).val,(execution p s).property,?_,?_⟩
  · intro c hc
    have hv := readback p s (target_injective m n b l t h s binding hp) (⟨c,hc⟩,PUnit.unit)
    simpa only [p,targetOffset_eq m n b l t h s binding hp,sourceOffset_eq] using hv
  · intro r o outside
    apply frame p s
    rcases outside with other | miss
    · exact Or.inl other
    · exact Or.inr (fun i => by rw [targetOffset_eq m n b l t h s binding hp];exact miss _ i.1.isLt)

#axiomsClean correctness
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.KernelReal
