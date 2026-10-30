import bench.tritonbench_g.var_len_copy.Proofs.IOLayout
import bench.tritonbench_g.var_len_copy.Proofs.KernelRealExecution
import bench.tritonbench_g.var_len_copy.Proofs.KernelRealSafetySource

namespace VeriTile.Bench.Repairs.VarLenCopy.PublicIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

theorem metadata_binding (a : Segment) (xs : (io a).Ctx) (hd : metadataDomain a xs)
    (s : BlockState) (binding : (io a).RegionalInputs s xs) :
    s.readMemValue .int "length" (s.pids 0)=a.length.toInt ∧
    (0<a.len → s.readMemValue .int "old" (s.pids 0)=(a.old : Int) ∧
      s.readMemValue .int "new" (s.pids 0)=(a.dest : Int)) := by
  refine ⟨(binding (⟨0,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1) trivial).trans hd.1,?_⟩
  intro positive
  exact ⟨(binding (⟨1,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1) trivial).trans (hd.2 positive).1,
    (binding (⟨2,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1) trivial).trans (hd.2 positive).2⟩

theorem source_safe (a : Segment) (s : BlockState) (xs : (io a).Ctx)
    (domain : metadataDomain a xs) (binding : (io a).RegionalInputs s xs)
    (bounds : RegionBounds) (hb : (io a).RegionalBounds bounds s xs) :
    (io a).kernel.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨length,starts⟩ := metadata_binding a xs domain s binding
  have hl : s.pids 0<bounds "length" := hb.1 (⟨0,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1) trivial
  have ho : s.pids 0<bounds "old" := hb.1 (⟨1,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1) trivial
  have hn : s.pids 0<bounds "new" := hb.1 (⟨2,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1) trivial
  by_cases positive : 0<a.len
  · let last : Fin a.len := ⟨a.len-1,by omega⟩
    have input : a.old+last.val<bounds "source" := hb.1 (⟨3,by decide⟩ : Fin 4) last trivial
    have output : a.dest+last.val<bounds "output" := hb.2.1 (⟨0,by decide⟩ : Fin 1) last trivial
    exact KernelReal.source_safe bounds s ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output"
      256 a.length.toInt a.old a.dest (by decide) (by decide) length (starts positive).1
      (starts positive).2 hl ho hn (by change a.old+a.len≤_; dsimp [last] at input; omega)
      (by change a.dest+a.len≤_; dsimp [last] at output; omega)
  · have nonpositive : a.length.toInt≤0 := by
      change ¬0<a.length.toInt.toNat at positive
      omega
    exact KernelReal.source_empty_safe bounds s ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output"
      256 a.length.toInt (by decide) nonpositive length hl ho hn

theorem source_result (a : Segment) (s : BlockState) (xs : (io a).Ctx)
    (domain : metadataDomain a xs) (binding : (io a).RegionalInputs s xs) :
    ∃ final, exec (io a).kernel.toAlgKernel s=some final ∧
      (∀ (o : Fin (io a).nOut) (j : Fin ((io a).oarity o)),
        (io a).omask o xs (s.pids 0) (s.pids 1) (s.pids 2) j →
        ((io a).oty o).read final ((io a).obuf o)
          ((io a).owin o xs (s.pids 0) (s.pids 1) (s.pids 2) j)=
            formula a (s.pids 0) (s.pids 1) (s.pids 2) xs o j) ∧
      (io a).RegionalFrame s final xs := by
  obtain ⟨length,starts⟩ := metadata_binding a xs domain s binding
  obtain ⟨final,executed,values,frame⟩ := RealExecution.var_len_copy_kernel_real a s length starts
  refine ⟨final,executed,?_,?_⟩
  · intro o j active
    exact (values j.val j.isLt).trans (binding (⟨3,by decide⟩ : Fin 4) j trivial)
  · intro r address outside _
    apply frame r address
    by_cases hr : r="output"
    · right
      by_cases before : address<a.dest
      · exact Or.inl before
      · right
        by_contra inside
        have hj : address-a.dest<a.len := by omega
        have miss := outside (⟨0,by decide⟩ : Fin 1) ⟨address-a.dest,hj⟩ trivial
        rcases miss with no | no
        · exact no hr
        · apply no
          change address=a.dest+(address-a.dest)
          omega
    · exact Or.inl hr

#axiomsClean metadata_binding
#axiomsClean source_safe
#axiomsClean source_result
end
end VeriTile.Bench.Repairs.VarLenCopy.PublicIO
