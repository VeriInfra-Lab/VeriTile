import bench.tritonbench_g.attn_fwd_triton.NativeIO
import bench.tritonbench_g.attn_fwd_triton.Proofs.ProgramSafety

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.NativeInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model ReadLocality
open AttnFwdTriton.Verified.PhysicalCells (Role)
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem channel_input (p : Config) (i : Fin (inputs p).length)
    (j : TileIndex (inputs p)[i].shape) : InputAddress p ((inputs p)[i].buf,read p i j) := by
  fin_cases i
  · exact ⟨j.1.val,j.2.1.val,j.2.2.1.val,j.2.2.2.1.val,.query,by decide,
      ⟨j.1.isLt,j.2.1.isLt,j.2.2.1.isLt,j.2.2.2.1.isLt⟩,rfl⟩
  · exact ⟨j.1.val,j.2.1.val,j.2.2.1.val,j.2.2.2.1.val,.key,by decide,
      ⟨j.1.isLt,j.2.1.isLt,j.2.2.1.isLt,j.2.2.2.1.isLt⟩,rfl⟩
  · exact ⟨j.1.val,j.2.1.val,j.2.2.1.val,j.2.2.2.1.val,.value,by decide,
      ⟨j.1.isLt,j.2.1.isLt,j.2.2.1.isLt,j.2.2.2.1.isLt⟩,rfl⟩
  · exact ⟨0,0,j.1.val,0,.queryScale,by decide,⟨rfl,rfl,rfl,j.1.isLt⟩,rfl⟩
  · exact ⟨0,0,j.1.val,0,.keyScale,by decide,⟨rfl,rfl,rfl,j.1.isLt⟩,rfl⟩

theorem input_location (p : Config) (key : RegionName × Nat) (h : InputAddress p key) :
    Nonempty (InputLocation p key.1 key.2) := by
  obtain ⟨z,h,n,d,r,hr,active,rfl⟩ := h
  cases r
  · exact ⟨⟨⟨0,by simp [inputs]⟩,
      (⟨z,active.1⟩,⟨h,active.2.1⟩,⟨n,active.2.2.1⟩,⟨d,active.2.2.2⟩,PUnit.unit),rfl,rfl⟩⟩
  · exact ⟨⟨⟨1,by simp [inputs]⟩,
      (⟨z,active.1⟩,⟨h,active.2.1⟩,⟨n,active.2.2.1⟩,⟨d,active.2.2.2⟩,PUnit.unit),rfl,rfl⟩⟩
  · exact ⟨⟨⟨2,by simp [inputs]⟩,
      (⟨z,active.1⟩,⟨h,active.2.1⟩,⟨n,active.2.2.1⟩,⟨d,active.2.2.2⟩,PUnit.unit),rfl,rfl⟩⟩
  · exact False.elim (hr rfl)
  · exact ⟨⟨⟨3,by simp [inputs]⟩,(⟨n,active.2.2.2⟩,PUnit.unit),rfl,rfl⟩⟩
  · exact ⟨⟨⟨4,by simp [inputs]⟩,(⟨n,active.2.2.2⟩,PUnit.unit),rfl,rfl⟩⟩

theorem payload_binding (p : Config) (values : Inputs p) (s : State Expr)
    (bound : (programIO p).InputBinding values s)
    (i : Fin (inputs p).length) (j : TileIndex (inputs p)[i].shape) :
    (s.mem (inputs p)[i].buf (read p i j)).read .real = payload p values i j := by
  have h := bound i j trivial
  fin_cases i <;> simpa only [programIO,memory,inputs,payload,ProgramIO.InputBinding,
    List.getElem_cons_zero,List.getElem_cons_succ,mapValue,id_eq] using h

theorem logical_agree (p : Config) (values : Inputs p) (s : State Expr)
    (bound : (programIO p).InputBinding values s) :
    Agree p s.mem (logicalMemory p values) := by
  intro key input
  have hasLocation := input_location p key input
  let location := Classical.choice hasLocation
  have chosen := Classical.choose_spec location.property
  have hb := payload_binding p values s bound location.val (Classical.choose location.property)
  have keyeq : ((inputs p)[location.val].buf,
      read p location.val (Classical.choose location.property))=key := Prod.ext chosen.1 chosen.2
  have eq := congrArg (reads s.mem) keyeq
  have result := eq.symm.trans hb
  simpa only [reads,logicalMemory,dif_pos hasLocation,Cell.read_same] using result

theorem windows (p : Config) (values : Inputs p) (bounds : RegionName → Nat)
    (backed : (programIO p).InBounds bounds values) : SourceSafety.Windows p bounds := by
  constructor
  · intro key input
    obtain ⟨loc⟩ := input_location p key input
    obtain ⟨j,region,address⟩ := loc.property
    have h := backed.2.1 loc.val j trivial
    change read p loc.val j < bounds (inputs p)[loc.val].buf at h
    rwa [region,address] at h
  · intro c
    exact backed.2.2.1 ⟨0,by simp [programIO,memory,outputs]⟩ (index p c) trivial

theorem frame (p : Config) (values : Inputs p) (s t : State Expr)
    (outside : ∀ r o, ¬Window p r o → t.mem r o=s.mem r o) :
    (programIO p).Frame values s t := by
  intro r off h
  apply outside
  rintro ⟨region,c,address⟩
  have hn := h.1 ⟨0,by simp [programIO,memory,outputs]⟩ (index p c) trivial
  exact hn.elim (fun h => h region) (fun h => h address)

def fromMemory (p : Config) (s : State Expr) : Inputs p := fun i j =>
  (s.mem (inputs p)[i].buf (read p i j)).read (inputs p)[i].dtype

theorem fromMemory_binding (p : Config) (s : State Expr) :
    (programIO p).InputBinding (fromMemory p s) s := by
  intro i j _
  fin_cases i <;> rfl

#axiomsClean logical_agree
#axiomsClean windows
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.NativeInterface
