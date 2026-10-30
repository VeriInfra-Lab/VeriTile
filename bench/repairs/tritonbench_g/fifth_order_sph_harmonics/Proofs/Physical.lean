import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.SourceBridge

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Physical
open VeriTile.Triton FP.Structural Packing Storage
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 64000

structure Buffers where
  coordinates : Nat
  upstream : Nat
  destination : Nat

def backed (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap) : State Word where
  mem region index:=.mk .real (read heap ((if region=coordinateRegion then b.coordinates
    else if region=upstreamRegion then b.upstream else b.destination)+4*index))
  regs _ _ _:=none
  pids _:=0
  numPids dimension:=if dimension=0 then Grid.blocks p else 1
  undef dtype _ _:=defaultValue dtype

theorem backed_output (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap) (i : Nat) :
    ((backed d p b heap).mem (Grid.output d) i).read .real=read heap (b.destination+4*i) := by
  cases d <;> simp [backed,Grid.output,outputRegion,gradientRegion,coordinateRegion,upstreamRegion,Cell.read_same]

theorem initial_image (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap) :
    image (Grid.output d) b.destination (SourceBridge.extent p d) heap (backed d p b heap).mem=heap :=
  image_initial _ _ _ _ _ (fun i _=>backed_output d p b heap i)

def execute (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Buffers)
    (order : List Nat) (heap : Heap) :=
  SourceBridge.run M d p b.destination order ⟨backed d p b heap,heap⟩

def launch (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap) :=
  execute M d p b (List.range (Grid.blocks p)) heap

theorem offset_bounded (p : Parameters) (d : Grid.Direction) (fit : p.columnOffset+11≤p.outputWidth)
    (r : Nat) (hr : r<p.rows) (c : Fin (Grid.channels d)) :
    Grid.offset p d r c.val<SourceBridge.extent p d := by
  have hc:=c.isLt
  cases d
  · exact (Grid.row_mask _ _ _ _ _ fit hc).mpr hr
  · simp only [Grid.offset,Grid.stride,Grid.first,SourceBridge.extent,Nat.add_zero]
    simp only [Grid.channels] at hc
    omega

theorem complete (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth) :
    Grid.launch M d p (backed d p b heap)=some (launch M d p b heap).logical ∧
    (∀ r,r<p.rows → ∀ c : Fin (Grid.channels d),∀ k : Fin 4,
      (launch M d p b heap).bytes (b.destination+4*Grid.offset p d r c.val+k.val)=
        Grid.value M d p (backed d p b heap) r c k) ∧
    (∀ q,(∀ r,r<p.rows → ∀ c : Fin (Grid.channels d),∀ k : Fin 4,
      q≠b.destination+4*Grid.offset p d r c.val+k.val) →
      (launch M d p b heap).bytes q=heap q) := by
  have projection:=SourceBridge.run_projection M d p b.destination (List.range (Grid.blocks p)) heap
    ⟨backed d p b heap,heap⟩ (initial_image d p b heap).symm
  obtain ⟨t,run,values,frame⟩:=Grid.launch_correct M d p (backed d p b heap) fit
  have eq : t=(launch M d p b heap).logical := Option.some.inj (run.symm.trans projection.1)
  subst t
  refine ⟨projection.1,?_,?_⟩
  · intro r hr c k
    rw [show (launch M d p b heap).bytes=image (Grid.output d) b.destination
      (SourceBridge.extent p d) heap (launch M d p b heap).logical.mem from projection.2]
    rw [image_byte _ _ _ _ _ _ (offset_bounded p d fit r hr c) k,values r hr c,Cell.read_same]
  · intro q outside
    rw [show (launch M d p b heap).bytes=image (Grid.output d) b.destination
      (SourceBridge.extent p d) heap (launch M d p b heap).logical.mem from projection.2]
    by_cases inside : b.destination≤q ∧ q<b.destination+4*SourceBridge.extent p d
    · let i:=(q-b.destination)/4
      let k:Fin 4:=⟨(q-b.destination)%4,Nat.mod_lt _ (by omega)⟩
      have hi:i<SourceBridge.extent p d := by dsimp [i];omega
      have address:b.destination+4*i+k.val=q := by dsimp [i,k];omega
      rw [←address,image_byte _ _ _ _ _ i hi k,frame]
      · rw [backed_output]
        rfl
      · right
        intro r hr c same
        exact outside r hr c k (by rw [←same];exact address.symm)
    · exact image_frame _ _ _ _ _ q (by omega)

/-- Source-state inputs and the evolving byte heap agree after any complete
launch whenever the destination stores are physically disjoint from that read. -/
theorem input_frame (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth) (address : Nat)
    (apart : ∀ k : Fin 4,∀ r,r<p.rows → ∀ c : Fin (Grid.channels d),∀ j : Fin 4,
      address+k.val≠b.destination+4*Grid.offset p d r c.val+j.val) :
    read (launch M d p b heap).bytes address=read heap address := by
  funext k
  exact (complete M d p b heap fit).2.2 (address+k.val) (apart k)

/-- At every program boundary, an immutable physical input still backs the
source interpreter's input region. This applies to every prefix of the launch. -/
theorem prefix_input (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap)
    (order : List Nat) (nodup : order.Nodup) (fit : p.columnOffset+11≤p.outputWidth)
    (region : RegionName) (index address : Nat) (different : region≠Grid.output d)
    (backing : ((backed d p b heap).mem region index).read .real=read heap address)
    (apart : ∀ k : Fin 4,address+k.val<b.destination ∨
      b.destination+4*SourceBridge.extent p d≤address+k.val) :
    ((execute M d p b order heap).logical.mem region index).read .real=
      read (execute M d p b order heap).bytes address := by
  have projection:=SourceBridge.run_projection M d p b.destination order heap
    ⟨backed d p b heap,heap⟩ (initial_image d p b heap).symm
  obtain ⟨t,run,_,frame⟩:=Grid.order_correct M d p order nodup (backed d p b heap) fit
  have eq:t=(execute M d p b order heap).logical := Option.some.inj (run.symm.trans projection.1)
  subst t
  rw [frame region index (Or.inl different),backing]
  funext k
  symm
  change (execute M d p b order heap).bytes (address+k.val)=heap (address+k.val)
  rw [show (execute M d p b order heap).bytes=image (Grid.output d) b.destination
    (SourceBridge.extent p d) heap (execute M d p b order heap).logical.mem from projection.2]
  exact image_frame _ _ _ _ _ _ (apart k)

theorem prefix_output (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Buffers) (heap : Heap)
    (order : List Nat) (i : Nat) (hi:i<SourceBridge.extent p d) :
    read (execute M d p b order heap).bytes (b.destination+4*i)=
      ((execute M d p b order heap).logical.mem (Grid.output d) i).read .real := by
  have projection:=SourceBridge.run_projection M d p b.destination order heap
    ⟨backed d p b heap,heap⟩ (initial_image d p b heap).symm
  funext k
  change (execute M d p b order heap).bytes (b.destination+4*i+k.val)=_
  rw [show (execute M d p b order heap).bytes=image (Grid.output d) b.destination
    (SourceBridge.extent p d) heap (execute M d p b order heap).logical.mem from projection.2]
  exact image_byte _ _ _ _ _ i hi k

#axiomsClean complete
#axiomsClean input_frame
#axiomsClean prefix_input
#axiomsClean prefix_output
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Physical
