import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.RealExecution

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.RealGrid
open VeriTile.Triton RealExecution
noncomputable section
set_option maxHeartbeats 4000000

abbrev address (h : Call) (p : Geometry.Program h) := Geometry.outputOffset h p.batch p.row 0

def launch (h : Call) (p : Geometry.Program h) (s : BlockState) : BlockState :=
  {s with regs := fun _ _ _ => none
          pids := fun ax => if ax=0 then p.row else if ax=1 then p.batch else 0
          numPids := fun ax => if ax=0 then h.N else if ax=1 then h.batches else 1}

def run (h : Call) : List (Geometry.Program h) → BlockState → Option BlockState
  | [],s => some s
  | p::rest,s => (exec (Kernels.source h).eraseDType.toAlgKernel (launch h p s)).bind (run h rest)

def Writes (h : Call) (indices : Nat → Int) (ps : List (Geometry.Program h)) (r : RegionName) (off : Nat) : Prop :=
  ∃ p∈ps,writes h p (indices p.batch) r off

theorem formula_congr (h : Call) (p : Geometry.Program h) (s t : BlockState) (slot : Nat)
    (x : ∀ off,t.mem "X" off=s.mem "X" off)
    (w : ∀ off,t.mem "W" off=s.mem "W" off)
    (y : t.mem "Y" (address h p)=s.mem "Y" (address h p)) :
    formula h p t slot=formula h p s slot := by
  simp only [formula,BlockState.readMem,x,w,y]

theorem complete (h : Call) (ps : List (Geometry.Program h))
    (unique : ps.Pairwise (fun p q => address h p≠address h q))
    (indices : Nat → Int) (valid : ∀ b,b<h.batches → indices b=-1 ∨ 0 ≤ indices b)
    (s : BlockState) (metadata : ∀ b,b<h.batches → s.readMemValue .int "Indices" b=indices b) :
    ∃ t,run h ps s=some t ∧
      (∀ p∈ps,indices p.batch≠-1 → t.readMem "Y" (address h p)=formula h p s (indices p.batch).toNat) ∧
      (∀ r off,¬Writes h indices ps r off → t.mem r off=s.mem r off) := by
  induction ps generalizing s with
  | nil => exact ⟨s,rfl,by simp,fun _ _ _ => rfl⟩
  | cons p rest ih =>
    obtain ⟨different,uniqueRest⟩ := List.pairwise_cons.mp unique
    obtain ⟨mid,hm,_,hv,hf⟩ := source_executes h p (launch h p s) rfl rfl
      (indices p.batch) (valid p.batch p.batchInside) (metadata p.batch p.batchInside)
    have frame (r : RegionName) (off : Nat) (outside : ¬writes h p (indices p.batch) r off) :
        mid.mem r off=s.mem r off := hf r off outside
    have metadataMid : ∀ b,b<h.batches → mid.readMemValue .int "Indices" b=indices b := by
      intro b hb
      simpa only [BlockState.readMemValue,BlockState.readMemTyped,frame "Indices" b (by simp [writes])] using metadata b hb
    obtain ⟨t,ht,values,restFrame⟩ := ih uniqueRest mid metadataMid
    refine ⟨t,by simp only [run,hm,Option.bind_some,ht],?_,?_⟩
    · intro q member active
      rcases List.mem_cons.mp member with equal | later
      · subst q
        have unchanged := restFrame "Y" (address h p) (by
          rintro ⟨q,hq,_,_,same⟩
          exact different q hq same)
        rw [BlockState.readMem,unchanged]
        simpa only [formula,launch,BlockState.readMem] using hv active
      · rw [values q later active]
        apply formula_congr
        · intro off; exact frame "X" off (by simp [writes])
        · intro off; exact frame "W" off (by simp [writes])
        · exact frame "Y" (address h q) (by
            rintro ⟨_,_,same⟩
            exact different q later same.symm)
    · intro r off outside
      rw [restFrame r off (fun ⟨q,hq,hw⟩ => outside ⟨q,by simp [hq],hw⟩)]
      exact frame r off (fun hw => outside ⟨p,by simp,hw⟩)

#axiomsClean complete
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.RealGrid
