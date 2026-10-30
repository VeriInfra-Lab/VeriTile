import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeReferenceConnection

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

@[simp] theorem patch_none (p : Parameters) (initial : Memory) :
    patch p initial (fun _=>none)=initial := by
  funext r off
  simp [patch]

theorem patch_update (p : Parameters) (initial : Memory) (writes : Writes) (off : Nat) (x : Expr) :
    NativeReference.writeMemory (patch p initial writes) p.Out off (.mk .fp32 x)=
      patch p initial (Function.update writes off (some x)) := by
  funext r addressIndex
  by_cases region : r=p.Out <;> by_cases address : addressIndex=off <;>
    simp [NativeReference.writeMemory,patch,region,address,Function.update_apply]

theorem tile_memory (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (pid : Host.PID (parameters p) c)
    (nonempty : p.shape.outputSize≠0) :
    NativeReference.evaluate (parameters p) c (coordinates p c pid) (patch p initial writes)=
      patch p initial (tile p c v pid writes) := by
  unfold NativeReference.evaluate tile
  split_ifs with active
  · unfold NativeReference.scatter
    have fold : ∀ (indices : List (TileIndex [c.M,c.N])) (current : Writes),
        indices.foldl (fun m i=>
          if SourceAccess.outMask (parameters p) c (coordinates p c pid) i.1 i.2.1 then
            NativeReference.writeMemory m p.Out
              (SourcePointers.outOffset (parameters p) c (coordinates p c pid) i.1 i.2.1)
              (.mk .fp32 (NativeReference.output (parameters p) c (coordinates p c pid)
                (NativeReference.capture (parameters p) (patch p initial writes)) i))
          else m) (patch p initial current)=
        patch p initial (indices.foldl (fun acc i=>
          if SourceAccess.outMask (parameters p) c (coordinates p c pid) i.1 i.2.1 then
            Function.update acc (SourcePointers.outOffset (parameters p) c (coordinates p c pid) i.1 i.2.1)
              (some (NativeReference.output (parameters p) c (coordinates p c pid) (operands p v writes) i))
          else acc) current) := by
      intro indices
      induction indices with
      | nil => intro current; rfl
      | cons i indices ih =>
        intro current
        simp only [List.foldl_cons]
        split_ifs with mask
        · rw [output_eq p c v initial reads writes pid nonempty active,patch_update]
          exact ih _
        · exact ih _
    simpa only [active,↓reduceIte] using fold (TileShape.allIndices [c.M,c.N]) writes
  · simp [active]

theorem grid_memory (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (nonempty : p.shape.outputSize≠0)
    (order : List (Host.PID (parameters p) c)) (writes : Writes) :
    NativeReference.grid (parameters p) c order (patch p initial writes)=
      patch p initial (order.foldl (fun current pid=>tile p c v pid current) writes) := by
  induction order generalizing writes with
  | nil => rfl
  | cons pid order ih =>
    change NativeReference.grid _ _ order
      (NativeReference.evaluate _ _ (coordinates p c pid) (patch p initial writes))=_
    rw [tile_memory p c v initial reads writes pid nonempty]
    exact ih _

theorem clear_memory (p : Parameters) (initial : Memory) (writes : Writes) (indices : List Nat) :
    (fun r off=>if r=p.Out then
      indices.foldl (Cleanup.store (Cell.mk .fp32 (.leaf .fp32 0))) (patch p initial writes r) off
      else patch p initial writes r off)=
    patch p initial (indices.foldl (fun current off=>Function.update current off (some (.leaf .fp32 0))) writes) := by
  funext r off
  by_cases region : r=p.Out
  · subst r
    simp only [↓reduceIte]
    have fold : ∀ indices : List Nat, ∀ writes : Writes,
        indices.foldl (Cleanup.store (Cell.mk .fp32 (.leaf .fp32 0))) (patch p initial writes p.Out)=
        patch p initial (indices.foldl (fun current off=>Function.update current off (some (.leaf .fp32 0))) writes) p.Out := by
      intro indices
      induction indices with
      | nil => intro writes; rfl
      | cons addressIndex indices ih =>
        intro writes
        simp only [List.foldl_cons]
        have update : Cleanup.store (Cell.mk .fp32 (.leaf .fp32 0)) (patch p initial writes p.Out) addressIndex=
            patch p initial (Function.update writes addressIndex (some (.leaf .fp32 0))) p.Out := by
          funext off
          by_cases eq : off=addressIndex <;> simp [Cleanup.store,patch,Function.update_apply,eq]
        rw [update,ih]
    exact congrFun (fold indices writes) off
  · simp [region,patch]

theorem kernel_memory (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) :
    NativeReference.kernelMemory (parameters p) c initial=patch p initial (complete p c v) := by
  by_cases empty : p.shape.outputSize=0
  · simpa only [NativeReference.kernelMemory,complete,show (parameters p).shape=p.shape from rfl,
      if_pos empty] using (patch_none p initial).symm
  · have main := grid_memory p c v initial reads empty (FPGrid.allPrograms (parameters p) c) (fun _=>none)
    rw [patch_none] at main
    simp only [NativeReference.kernelMemory,complete,show (parameters p).shape=p.shape from rfl,
      if_neg empty,main]
    unfold NativeReference.clearMemory
    cases causal : p.causal
    · simp only [show (parameters p).causal=p.causal from rfl,causal,Bool.false_eq_true,↓reduceIte]
      rfl
    · simpa only [show (parameters p).causal=p.causal from rfl,causal,↓reduceIte,Cleanup.run,ofFloat] using
        clear_memory p initial (grid p c v) (Cleanup.addresses (parameters p))

theorem write_bound (p : Parameters) (c : Host.Config) (off : Nat)
    (written : writeMask p c off) : off<p.shape.outputSize := by
  rcases written with ⟨_,(⟨pid,active,i,j,mask,rfl⟩|⟨causal,upper⟩)⟩
  · exact output_address_bound p c pid active i j mask
  · obtain ⟨pid,i,j,active,eq⟩ := (Cleanup.member (parameters p) off).mp ((Cleanup.image (parameters p) off).mpr upper)
    rw [eq]
    exact Cleanup.offset_bound (parameters p) pid i j active

theorem tile_frame (p : Parameters) (c : Host.Config) (v : Values p)
    (pid : Host.PID (parameters p) c) (writes : Writes) (off : Nat)
    (nonempty : p.shape.outputSize≠0) (outside : ¬writeMask p c off) :
    tile p c v pid writes off=writes off := by
  unfold tile
  split_ifs with active
  · have fold : ∀ (indices : List (TileIndex [c.M,c.N])) (current : Writes),
        indices.foldl (fun acc i=>
          if SourceAccess.outMask (parameters p) c (coordinates p c pid) i.1 i.2.1 then
            Function.update acc (SourcePointers.outOffset (parameters p) c (coordinates p c pid) i.1 i.2.1)
              (some (NativeReference.output (parameters p) c (coordinates p c pid) (operands p v writes) i))
          else acc) current off=current off := by
      intro indices
      induction indices with
      | nil => intro current; rfl
      | cons i indices ih =>
        intro current
        simp only [List.foldl_cons]
        split_ifs with mask
        · rw [ih,Function.update_of_ne]
          intro eq
          exact outside ⟨nonempty,Or.inl ⟨pid,active,i.1,i.2.1,mask,eq.symm⟩⟩
        · exact ih _
    exact fold _ _
  · rfl

theorem complete_frame (p : Parameters) (c : Host.Config) (v : Values p) (off : Nat)
    (outside : ¬writeMask p c off) : complete p c v off=none := by
  unfold complete
  split_ifs with empty causal
  · rfl
  · have noCleanup : off∉Cleanup.addresses (parameters p) := by
      intro member
      exact outside ⟨empty,Or.inr ⟨causal,(Cleanup.image (parameters p) off).mp member⟩⟩
    change (Cleanup.addresses (parameters p)).foldl (Cleanup.store (some (Expr.leaf .fp32 0))) (grid p c v) off=none
    rw [Cleanup.fold_frame (some (Expr.leaf .fp32 0)) _ _ off noCleanup]
    unfold grid
    have fold : ∀ (order : List (Host.PID (parameters p) c)) (current : Writes),
        order.foldl (fun current pid=>tile p c v pid current) current off=current off := by
      intro order
      induction order with
      | nil => intro current; rfl
      | cons pid rest ih => intro current; rw [List.foldl_cons,ih,tile_frame p c v pid current off empty outside]
    exact fold _ _
  · unfold grid
    have fold : ∀ (order : List (Host.PID (parameters p) c)) (current : Writes),
        order.foldl (fun current pid=>tile p c v pid current) current off=current off := by
      intro order
      induction order with
      | nil => intro current; rfl
      | cons pid rest ih => intro current; rw [List.foldl_cons,ih,tile_frame p c v pid current off empty outside]
    exact fold _ _

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
