import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOEvents

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOCoverage
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def index (fp8 : Bool) (p : Fin 3 → Arguments) (e : IO.Event order) :
    TileIndex (IO.outputs fp8 p order)[e.phase].shape :=
  cast (congrArg TileIndex (congrArg IOChannel.shape (IO.output_channel fp8 p order e.phase)).symm)
    (e.launch,e.lane)

@[simp] theorem index_roundtrip (fp8 : Bool) (p : Fin 3 → Arguments) (e : IO.Event order) :
    IO.outputIndex fp8 p order e.phase (index fp8 p e)=(e.launch,e.lane) := by
  simp [index,IO.outputIndex]

theorem outside_none (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (r : RegionName) (off : Nat)
    (outside : (IO.qkvIO a native fp8 p epsilon theta order).Outside vals r off) :
    IO.lastAt p order r off=none := by
  cases h : IO.lastAt p order r off with
  | none=>rfl
  | some e=>
    have sound:=IOEvents.last_sound p order r off e h
    have winner : IO.Winner p e := by simpa only [IO.Winner,sound.2.1,sound.2.2] using h
    have miss:=outside.1 e.phase (index fp8 p e)
    have mask : (IO.qkvIO a native fp8 p epsilon theta order).outputMask e.phase vals (index fp8 p e) := by
      simpa only [IO.qkvIO,id_eq,index_roundtrip] using winner
    have hm:=miss mask
    simp only [IO.qkvIO,id_eq,IO.output_channel,index_roundtrip] at hm
    rcases e with ⟨launch,phase,lane⟩
    fin_cases phase <;>
      exact False.elim (hm.elim (fun bad=>bad sound.2.1.symm) (fun bad=>bad sound.2.2.symm))

theorem write_bound (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (bounds : RegionName → Nat)
    (bound : (IO.qkvIO a native fp8 p epsilon theta order).InBounds bounds vals)
    (e : IO.Event order) (active : IO.eventActive p e) :
    IO.eventOffset p e<bounds (IO.eventRegion p e) := by
  obtain ⟨winner,hw,hr,ho⟩:=IOEvents.covered p e active
  have mask : (IO.qkvIO a native fp8 p epsilon theta order).outputMask winner.phase vals (index fp8 p winner) := by
    simpa only [IO.qkvIO,id_eq,index_roundtrip] using hw
  have hb:=bound.2.2.1 winner.phase (index fp8 p winner) mask
  simp only [IO.qkvIO,id_eq,index_roundtrip,IO.output_channel] at hb
  have hb' : IO.eventOffset p winner<bounds (IO.eventRegion p winner) := by
    rcases winner with ⟨launch,phase,lane⟩
    fin_cases phase <;> exact hb
  simpa only [hr,ho] using hb'

theorem x_bound (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (bounds : RegionName → Nat)
    (bound : (IO.qkvIO a native fp8 p epsilon theta order).InBounds bounds vals)
    (phase : Fin 3) (pid : Nat → Nat) (member : pid∈order) (m : Fin 16) (k : Nat) (hk : k<(p phase).K) :
    pid 0*(p phase).sxb+((pid 1/columns (p phase)*16+m.val)%(p phase).M)*(p phase).sxm+k*(p phase).sxk
      <bounds (p phase).X := by
  have hm : pid 0*(p phase).sxb+((pid 1/columns (p phase)*16+m.val)%(p phase).M)*(p phase).sxm+k*(p phase).sxk ∈ IO.xReads (p phase) order := by
    apply List.mem_flatMap.mpr
    refine ⟨pid,member,?_⟩
    apply List.mem_flatMap.mpr
    exact ⟨m.val,List.mem_range.mpr m.isLt,List.mem_map.mpr ⟨k,List.mem_range.mpr hk,rfl⟩⟩
  have hb:=bound.2.1 (IO.xSlot phase)
  fin_cases phase <;> exact hb _ hm

theorem w_bound (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (bounds : RegionName → Nat)
    (bound : (IO.qkvIO a native fp8 p epsilon theta order).InBounds bounds vals)
    (phase : Fin 3) (pid : Nat → Nat) (member : pid∈order) (n : Fin 64) (k : Nat) (hk : k<(p phase).K) :
    k*(p phase).swk+((pid 1%columns (p phase)*64+n.val)%(p phase).N)*(p phase).swn<bounds (p phase).W := by
  have hm : k*(p phase).swk+((pid 1%columns (p phase)*64+n.val)%(p phase).N)*(p phase).swn ∈ IO.wReads (p phase) order := by
    apply List.mem_flatMap.mpr
    refine ⟨pid,member,?_⟩
    apply List.mem_flatMap.mpr
    exact ⟨n.val,List.mem_range.mpr n.isLt,List.mem_map.mpr ⟨k,List.mem_range.mpr hk,rfl⟩⟩
  have hb:=bound.2.1 (IO.wSlot phase)
  fin_cases phase <;> exact hb _ hm

theorem rms_bound (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (bounds : RegionName → Nat)
    (bound : (IO.qkvIO a native fp8 p epsilon theta order).InBounds bounds vals)
    (phase : Fin 3) (pid : Nat → Nat) (member : pid∈order) (k : Nat) (hk : k<(p phase).K) :
    k*(p phase).srms<bounds (p phase).RMS := by
  have hm : k*(p phase).srms ∈ IO.rmsReads (p phase) order := by
    exact List.mem_flatMap.mpr ⟨pid,member,List.mem_map.mpr ⟨k,List.mem_range.mpr hk,rfl⟩⟩
  have hb:=bound.2.1 (IO.rmsSlot phase)
  fin_cases phase <;> exact hb _ hm

#axiomsClean outside_none
#axiomsClean write_bound
#axiomsClean x_bound
#axiomsClean w_bound
#axiomsClean rms_bound
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOCoverage
