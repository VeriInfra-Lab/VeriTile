import bench.tritonbench_g.parallel_attention.Proofs.IOSchedule

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option linter.unusedSimpArgs false

@[simp] theorem output_address (shape : Shape) (j : Point shape 128 shape.BV) :
    (outputRectangle shape (program shape 128 shape.BV j).tile 128).address
      ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩=outputWindow shape (⟨0,by decide⟩ : Fin 5) j := by
  simp [outputRectangle,program,Program.tile,Rectangle.address,outputWindow,dense,row,col,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

@[simp] theorem normalizer_address (shape : Shape) (j : NormPoint shape) :
    normalizerAddress shape (normProgram shape j).tile j.2.2.2.1.val=outputWindow shape (⟨1,by decide⟩ : Fin 5) j := by
  simp [normalizerAddress,normProgram,Program.tile,outputWindow,Nat.add_assoc]

@[simp] theorem query_address (shape : Shape) (j : Point shape 64 shape.BK) :
    (queryRectangle shape (program shape 64 shape.BK j).tile).address
      ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩=outputWindow shape (⟨2,by decide⟩ : Fin 5) j := by
  simp [queryRectangle,program,Program.tile,Rectangle.address,outputWindow,dense,row,heads,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

@[simp] theorem key_address (shape : Shape) (j : Point shape 64 shape.BK) :
    (keyRectangle shape (program shape 64 shape.BK j).tile).address
      ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩=outputWindow shape (⟨3,by decide⟩ : Fin 5) j := query_address shape j

@[simp] theorem value_address (shape : Shape) (j : Point shape 64 shape.BV) :
    (valueRectangle shape (program shape 64 shape.BV j).tile).address
      ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩=outputWindow shape (⟨4,by decide⟩ : Fin 5) j := by
  simp [valueRectangle,outputRectangle,program,Program.tile,Rectangle.address,outputWindow,dense,row,col,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem forward_outside {α : Type} (shape : Shape) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (R : RegionName) (off : Nat)
    (h : (dataIO α .forward shape jobs).Outside xs R off) (p : Program shape 128) :
    ¬forwardWrites shape p R off := by
  intro writes
  rcases writes with ⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩|⟨writer,rfl,⟨r,active,rfl⟩⟩
  · have active' : outputActive .forward shape 0 (p.head,p.chunk,p.value,r,v,()) := ⟨rfl,active⟩
    have outside:=h.1 (⟨0,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,r,v,()) active'
    exact outside.elim (fun e=>e rfl) (fun e=>e (output_address shape (p.head,p.chunk,p.value,r,v,())))
  · have active' : outputActive .forward shape 1 (p.head,p.chunk,p.value,r,()) := ⟨rfl,writer,active⟩
    have outside:=h.1 (⟨1,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,r,()) active'
    exact outside.elim (fun e=>e rfl) (fun e=>e (normalizer_address shape (p.head,p.chunk,p.value,r,())))

theorem backward_outside {α : Type} (shape : Shape) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (R : RegionName) (off : Nat)
    (h : (dataIO α .backward shape jobs).Outside xs R off) (p : Program shape 64) :
    ¬backwardWrites shape p R off := by
  intro writes
  rcases writes with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|
    ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩
  · have active' : outputActive .backward shape 2 (p.head,p.chunk,p.value,r,e,()) := ⟨rfl,by simpa [queryRectangle,Rectangle.active,Program.tile,row] using active⟩
    have outside:=h.1 (⟨2,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,r,e,()) active'
    exact outside.elim (fun e=>e rfl) (fun bad=>bad (query_address shape (p.head,p.chunk,p.value,r,e,())))
  · have active' : outputActive .backward shape 3 (p.head,p.chunk,p.value,r,e,()) := ⟨rfl,by simpa [keyRectangle,queryRectangle,Rectangle.active,Program.tile,row] using active⟩
    have outside:=h.1 (⟨3,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,r,e,()) active'
    exact outside.elim (fun e=>e rfl) (fun bad=>bad (key_address shape (p.head,p.chunk,p.value,r,e,())))
  · have active' : outputActive .backward shape 4 (p.head,p.chunk,p.value,r,v,()) := ⟨rfl,active⟩
    have outside:=h.1 (⟨4,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,r,v,()) active'
    exact outside.elim (fun e=>e rfl) (fun e=>e (value_address shape (p.head,p.chunk,p.value,r,v,())))

#axiomsClean forward_outside
#axiomsClean backward_outside
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
