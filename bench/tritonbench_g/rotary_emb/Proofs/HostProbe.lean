import bench.tritonbench_g.rotary_emb.Proofs.HostMetadata

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
open VeriTile.Triton

def probe0 : Request :=
  ⟨⟨17,5,0,1000,23,2⟩,⟨17,10,3,2000,31,3⟩,
    ⟨17,0,300,4⟩,⟨17,1,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (0)) probe0 =
    some ⟨probe0,0⟩ := rfl
example : (Plan.mk probe0 (0)).grid.dims = [2,2] ∧
    (Plan.mk probe0 (0)).numWarps = 4 ∧
    (Plan.mk probe0 (0)).numStages = 1 := by decide

def probe1 : Request :=
  ⟨⟨17,5,1,1000,23,2⟩,⟨17,10,4,2000,31,3⟩,
    ⟨17,1,300,4⟩,⟨17,2,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (1)) probe1 =
    some ⟨probe1,1⟩ := rfl
example : (Plan.mk probe1 (1)).grid.dims = [2,2] ∧
    (Plan.mk probe1 (1)).numWarps = 4 ∧
    (Plan.mk probe1 (1)).numStages = 1 := by decide

def probe2 : Request :=
  ⟨⟨17,5,3,1000,23,2⟩,⟨17,10,6,2000,31,3⟩,
    ⟨17,3,300,4⟩,⟨17,4,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (3)) probe2 =
    some ⟨probe2,3⟩ := rfl
example : (Plan.mk probe2 (3)).grid.dims = [2,2] ∧
    (Plan.mk probe2 (3)).numWarps = 4 ∧
    (Plan.mk probe2 (3)).numStages = 1 := by decide

def probe3 : Request :=
  ⟨⟨17,5,7,1000,23,2⟩,⟨17,10,10,2000,31,3⟩,
    ⟨17,7,300,4⟩,⟨17,8,400,5⟩,0.75⟩
example : prepare (fun _ _ => some (5)) probe3 =
    some ⟨probe3,5⟩ := rfl
example : (Plan.mk probe3 (5)).grid.dims = [2,2] ∧
    (Plan.mk probe3 (5)).numWarps = 4 ∧
    (Plan.mk probe3 (5)).numStages = 1 := by decide

def probe4 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,0.5⟩
example : prepare (fun _ _ => some (32)) probe4 =
    some ⟨probe4,32⟩ := rfl
example : (Plan.mk probe4 (32)).grid.dims = [2,2] ∧
    (Plan.mk probe4 (32)).numWarps = 4 ∧
    (Plan.mk probe4 (32)).numStages = 1 := by decide

def probe5 : Request :=
  ⟨⟨17,5,127,1000,23,2⟩,⟨17,10,130,2000,31,3⟩,
    ⟨17,127,300,4⟩,⟨17,128,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (127)) probe5 =
    some ⟨probe5,127⟩ := rfl
example : (Plan.mk probe5 (127)).grid.dims = [2,2] ∧
    (Plan.mk probe5 (127)).numWarps = 4 ∧
    (Plan.mk probe5 (127)).numStages = 1 := by decide

def probe6 : Request :=
  ⟨⟨17,5,128,1000,23,2⟩,⟨17,10,131,2000,31,3⟩,
    ⟨17,128,300,4⟩,⟨17,129,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (128)) probe6 =
    some ⟨probe6,128⟩ := rfl
example : (Plan.mk probe6 (128)).grid.dims = [2,2] ∧
    (Plan.mk probe6 (128)).numWarps = 8 ∧
    (Plan.mk probe6 (128)).numStages = 1 := by decide

def probe7 : Request :=
  ⟨⟨17,5,129,1000,23,2⟩,⟨17,10,132,2000,31,3⟩,
    ⟨17,129,300,4⟩,⟨17,130,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (129)) probe7 =
    some ⟨probe7,129⟩ := rfl
example : (Plan.mk probe7 (129)).grid.dims = [2,2] ∧
    (Plan.mk probe7 (129)).numWarps = 8 ∧
    (Plan.mk probe7 (129)).numStages = 1 := by decide

def probe8 : Request :=
  ⟨⟨17,5,130,1000,23,2⟩,⟨17,10,133,2000,31,3⟩,
    ⟨17,130,300,4⟩,⟨17,131,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (130)) probe8 =
    some ⟨probe8,130⟩ := rfl
example : (Plan.mk probe8 (130)).grid.dims = [2,2] ∧
    (Plan.mk probe8 (130)).numWarps = 8 ∧
    (Plan.mk probe8 (130)).numStages = 1 := by decide

def probe9 : Request :=
  ⟨⟨17,5,256,1000,23,2⟩,⟨17,10,259,2000,31,3⟩,
    ⟨17,256,300,4⟩,⟨17,257,400,5⟩,0.5⟩
example : prepare (fun _ _ => some (128)) probe9 =
    some ⟨probe9,128⟩ := rfl
example : (Plan.mk probe9 (128)).grid.dims = [2,2] ∧
    (Plan.mk probe9 (128)).numWarps = 8 ∧
    (Plan.mk probe9 (128)).numStages = 1 := by decide

def probe10 : Request :=
  ⟨⟨17,5,256,1000,23,2⟩,⟨17,10,259,2000,31,3⟩,
    ⟨17,256,300,4⟩,⟨17,257,400,5⟩,0.499⟩
example : prepare (fun _ _ => some (127)) probe10 =
    some ⟨probe10,127⟩ := rfl
example : (Plan.mk probe10 (127)).grid.dims = [2,2] ∧
    (Plan.mk probe10 (127)).numWarps = 4 ∧
    (Plan.mk probe10 (127)).numStages = 1 := by decide

def probe11 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,0.0⟩
example : prepare (fun _ _ => some (0)) probe11 =
    some ⟨probe11,0⟩ := rfl
example : (Plan.mk probe11 (0)).grid.dims = [2,2] ∧
    (Plan.mk probe11 (0)).numWarps = 4 ∧
    (Plan.mk probe11 (0)).numStages = 1 := by decide

def probe12 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,-1.0⟩
example : prepare (fun _ _ => some (-64)) probe12 =
    some ⟨probe12,-64⟩ := rfl
example : (Plan.mk probe12 (-64)).grid.dims = [2,2] ∧
    (Plan.mk probe12 (-64)).numWarps = 4 ∧
    (Plan.mk probe12 (-64)).numStages = 1 := by decide

def probe13 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.5⟩
example : prepare (fun _ _ => some (96)) probe13 =
    some ⟨probe13,96⟩ := rfl
example : (Plan.mk probe13 (96)).grid.dims = [2,2] ∧
    (Plan.mk probe13 (96)).numWarps = 4 ∧
    (Plan.mk probe13 (96)).numStages = 1 := by decide

def probe17 : Request :=
  ⟨⟨0,0,64,1000,23,2⟩,⟨0,7,67,2000,31,3⟩,
    ⟨0,64,300,4⟩,⟨0,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe17 =
    some ⟨probe17,64⟩ := rfl
example : (Plan.mk probe17 (64)).grid.dims = [0,0] ∧
    (Plan.mk probe17 (64)).numWarps = 4 ∧
    (Plan.mk probe17 (64)).numStages = 1 := by decide

def probe18 : Request :=
  ⟨⟨0,1,64,1000,23,2⟩,⟨0,8,67,2000,31,3⟩,
    ⟨0,64,300,4⟩,⟨0,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe18 =
    some ⟨probe18,64⟩ := rfl
example : (Plan.mk probe18 (64)).grid.dims = [1,0] ∧
    (Plan.mk probe18 (64)).numWarps = 4 ∧
    (Plan.mk probe18 (64)).numStages = 1 := by decide

def probe19 : Request :=
  ⟨⟨0,4,64,1000,23,2⟩,⟨0,11,67,2000,31,3⟩,
    ⟨0,64,300,4⟩,⟨0,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe19 =
    some ⟨probe19,64⟩ := rfl
example : (Plan.mk probe19 (64)).grid.dims = [1,0] ∧
    (Plan.mk probe19 (64)).numWarps = 4 ∧
    (Plan.mk probe19 (64)).numStages = 1 := by decide

def probe20 : Request :=
  ⟨⟨0,5,64,1000,23,2⟩,⟨0,12,67,2000,31,3⟩,
    ⟨0,64,300,4⟩,⟨0,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe20 =
    some ⟨probe20,64⟩ := rfl
example : (Plan.mk probe20 (64)).grid.dims = [2,0] ∧
    (Plan.mk probe20 (64)).numWarps = 4 ∧
    (Plan.mk probe20 (64)).numStages = 1 := by decide

def probe21 : Request :=
  ⟨⟨0,8,64,1000,23,2⟩,⟨0,15,67,2000,31,3⟩,
    ⟨0,64,300,4⟩,⟨0,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe21 =
    some ⟨probe21,64⟩ := rfl
example : (Plan.mk probe21 (64)).grid.dims = [2,0] ∧
    (Plan.mk probe21 (64)).numWarps = 4 ∧
    (Plan.mk probe21 (64)).numStages = 1 := by decide

def probe22 : Request :=
  ⟨⟨1,0,64,1000,23,2⟩,⟨1,7,67,2000,31,3⟩,
    ⟨1,64,300,4⟩,⟨1,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe22 =
    some ⟨probe22,64⟩ := rfl
example : (Plan.mk probe22 (64)).grid.dims = [0,1] ∧
    (Plan.mk probe22 (64)).numWarps = 4 ∧
    (Plan.mk probe22 (64)).numStages = 1 := by decide

def probe23 : Request :=
  ⟨⟨1,1,64,1000,23,2⟩,⟨1,8,67,2000,31,3⟩,
    ⟨1,64,300,4⟩,⟨1,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe23 =
    some ⟨probe23,64⟩ := rfl
example : (Plan.mk probe23 (64)).grid.dims = [1,1] ∧
    (Plan.mk probe23 (64)).numWarps = 4 ∧
    (Plan.mk probe23 (64)).numStages = 1 := by decide

def probe24 : Request :=
  ⟨⟨1,4,64,1000,23,2⟩,⟨1,11,67,2000,31,3⟩,
    ⟨1,64,300,4⟩,⟨1,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe24 =
    some ⟨probe24,64⟩ := rfl
example : (Plan.mk probe24 (64)).grid.dims = [1,1] ∧
    (Plan.mk probe24 (64)).numWarps = 4 ∧
    (Plan.mk probe24 (64)).numStages = 1 := by decide

def probe25 : Request :=
  ⟨⟨1,5,64,1000,23,2⟩,⟨1,12,67,2000,31,3⟩,
    ⟨1,64,300,4⟩,⟨1,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe25 =
    some ⟨probe25,64⟩ := rfl
example : (Plan.mk probe25 (64)).grid.dims = [2,1] ∧
    (Plan.mk probe25 (64)).numWarps = 4 ∧
    (Plan.mk probe25 (64)).numStages = 1 := by decide

def probe26 : Request :=
  ⟨⟨1,8,64,1000,23,2⟩,⟨1,15,67,2000,31,3⟩,
    ⟨1,64,300,4⟩,⟨1,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe26 =
    some ⟨probe26,64⟩ := rfl
example : (Plan.mk probe26 (64)).grid.dims = [2,1] ∧
    (Plan.mk probe26 (64)).numWarps = 4 ∧
    (Plan.mk probe26 (64)).numStages = 1 := by decide

def probe27 : Request :=
  ⟨⟨15,0,64,1000,23,2⟩,⟨15,7,67,2000,31,3⟩,
    ⟨15,64,300,4⟩,⟨15,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe27 =
    some ⟨probe27,64⟩ := rfl
example : (Plan.mk probe27 (64)).grid.dims = [0,1] ∧
    (Plan.mk probe27 (64)).numWarps = 4 ∧
    (Plan.mk probe27 (64)).numStages = 1 := by decide

def probe28 : Request :=
  ⟨⟨15,1,64,1000,23,2⟩,⟨15,8,67,2000,31,3⟩,
    ⟨15,64,300,4⟩,⟨15,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe28 =
    some ⟨probe28,64⟩ := rfl
example : (Plan.mk probe28 (64)).grid.dims = [1,1] ∧
    (Plan.mk probe28 (64)).numWarps = 4 ∧
    (Plan.mk probe28 (64)).numStages = 1 := by decide

def probe29 : Request :=
  ⟨⟨15,4,64,1000,23,2⟩,⟨15,11,67,2000,31,3⟩,
    ⟨15,64,300,4⟩,⟨15,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe29 =
    some ⟨probe29,64⟩ := rfl
example : (Plan.mk probe29 (64)).grid.dims = [1,1] ∧
    (Plan.mk probe29 (64)).numWarps = 4 ∧
    (Plan.mk probe29 (64)).numStages = 1 := by decide

def probe30 : Request :=
  ⟨⟨15,5,64,1000,23,2⟩,⟨15,12,67,2000,31,3⟩,
    ⟨15,64,300,4⟩,⟨15,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe30 =
    some ⟨probe30,64⟩ := rfl
example : (Plan.mk probe30 (64)).grid.dims = [2,1] ∧
    (Plan.mk probe30 (64)).numWarps = 4 ∧
    (Plan.mk probe30 (64)).numStages = 1 := by decide

def probe31 : Request :=
  ⟨⟨15,8,64,1000,23,2⟩,⟨15,15,67,2000,31,3⟩,
    ⟨15,64,300,4⟩,⟨15,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe31 =
    some ⟨probe31,64⟩ := rfl
example : (Plan.mk probe31 (64)).grid.dims = [2,1] ∧
    (Plan.mk probe31 (64)).numWarps = 4 ∧
    (Plan.mk probe31 (64)).numStages = 1 := by decide

def probe32 : Request :=
  ⟨⟨16,0,64,1000,23,2⟩,⟨16,7,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe32 =
    some ⟨probe32,64⟩ := rfl
example : (Plan.mk probe32 (64)).grid.dims = [0,1] ∧
    (Plan.mk probe32 (64)).numWarps = 4 ∧
    (Plan.mk probe32 (64)).numStages = 1 := by decide

def probe33 : Request :=
  ⟨⟨16,1,64,1000,23,2⟩,⟨16,8,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe33 =
    some ⟨probe33,64⟩ := rfl
example : (Plan.mk probe33 (64)).grid.dims = [1,1] ∧
    (Plan.mk probe33 (64)).numWarps = 4 ∧
    (Plan.mk probe33 (64)).numStages = 1 := by decide

def probe34 : Request :=
  ⟨⟨16,4,64,1000,23,2⟩,⟨16,11,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe34 =
    some ⟨probe34,64⟩ := rfl
example : (Plan.mk probe34 (64)).grid.dims = [1,1] ∧
    (Plan.mk probe34 (64)).numWarps = 4 ∧
    (Plan.mk probe34 (64)).numStages = 1 := by decide

def probe35 : Request :=
  ⟨⟨16,5,64,1000,23,2⟩,⟨16,12,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe35 =
    some ⟨probe35,64⟩ := rfl
example : (Plan.mk probe35 (64)).grid.dims = [2,1] ∧
    (Plan.mk probe35 (64)).numWarps = 4 ∧
    (Plan.mk probe35 (64)).numStages = 1 := by decide

def probe36 : Request :=
  ⟨⟨16,8,64,1000,23,2⟩,⟨16,15,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe36 =
    some ⟨probe36,64⟩ := rfl
example : (Plan.mk probe36 (64)).grid.dims = [2,1] ∧
    (Plan.mk probe36 (64)).numWarps = 4 ∧
    (Plan.mk probe36 (64)).numStages = 1 := by decide

def probe37 : Request :=
  ⟨⟨17,0,64,1000,23,2⟩,⟨17,7,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe37 =
    some ⟨probe37,64⟩ := rfl
example : (Plan.mk probe37 (64)).grid.dims = [0,2] ∧
    (Plan.mk probe37 (64)).numWarps = 4 ∧
    (Plan.mk probe37 (64)).numStages = 1 := by decide

def probe38 : Request :=
  ⟨⟨17,1,64,1000,23,2⟩,⟨17,8,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe38 =
    some ⟨probe38,64⟩ := rfl
example : (Plan.mk probe38 (64)).grid.dims = [1,2] ∧
    (Plan.mk probe38 (64)).numWarps = 4 ∧
    (Plan.mk probe38 (64)).numStages = 1 := by decide

def probe39 : Request :=
  ⟨⟨17,4,64,1000,23,2⟩,⟨17,11,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe39 =
    some ⟨probe39,64⟩ := rfl
example : (Plan.mk probe39 (64)).grid.dims = [1,2] ∧
    (Plan.mk probe39 (64)).numWarps = 4 ∧
    (Plan.mk probe39 (64)).numStages = 1 := by decide

def probe40 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,12,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe40 =
    some ⟨probe40,64⟩ := rfl
example : (Plan.mk probe40 (64)).grid.dims = [2,2] ∧
    (Plan.mk probe40 (64)).numWarps = 4 ∧
    (Plan.mk probe40 (64)).numStages = 1 := by decide

def probe41 : Request :=
  ⟨⟨17,8,64,1000,23,2⟩,⟨17,15,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe41 =
    some ⟨probe41,64⟩ := rfl
example : (Plan.mk probe41 (64)).grid.dims = [2,2] ∧
    (Plan.mk probe41 (64)).numWarps = 4 ∧
    (Plan.mk probe41 (64)).numStages = 1 := by decide

def probe42 : Request :=
  ⟨⟨31,0,64,1000,23,2⟩,⟨31,7,67,2000,31,3⟩,
    ⟨31,64,300,4⟩,⟨31,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe42 =
    some ⟨probe42,64⟩ := rfl
example : (Plan.mk probe42 (64)).grid.dims = [0,2] ∧
    (Plan.mk probe42 (64)).numWarps = 4 ∧
    (Plan.mk probe42 (64)).numStages = 1 := by decide

def probe43 : Request :=
  ⟨⟨31,1,64,1000,23,2⟩,⟨31,8,67,2000,31,3⟩,
    ⟨31,64,300,4⟩,⟨31,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe43 =
    some ⟨probe43,64⟩ := rfl
example : (Plan.mk probe43 (64)).grid.dims = [1,2] ∧
    (Plan.mk probe43 (64)).numWarps = 4 ∧
    (Plan.mk probe43 (64)).numStages = 1 := by decide

def probe44 : Request :=
  ⟨⟨31,4,64,1000,23,2⟩,⟨31,11,67,2000,31,3⟩,
    ⟨31,64,300,4⟩,⟨31,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe44 =
    some ⟨probe44,64⟩ := rfl
example : (Plan.mk probe44 (64)).grid.dims = [1,2] ∧
    (Plan.mk probe44 (64)).numWarps = 4 ∧
    (Plan.mk probe44 (64)).numStages = 1 := by decide

def probe45 : Request :=
  ⟨⟨31,5,64,1000,23,2⟩,⟨31,12,67,2000,31,3⟩,
    ⟨31,64,300,4⟩,⟨31,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe45 =
    some ⟨probe45,64⟩ := rfl
example : (Plan.mk probe45 (64)).grid.dims = [2,2] ∧
    (Plan.mk probe45 (64)).numWarps = 4 ∧
    (Plan.mk probe45 (64)).numStages = 1 := by decide

def probe46 : Request :=
  ⟨⟨31,8,64,1000,23,2⟩,⟨31,15,67,2000,31,3⟩,
    ⟨31,64,300,4⟩,⟨31,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe46 =
    some ⟨probe46,64⟩ := rfl
example : (Plan.mk probe46 (64)).grid.dims = [2,2] ∧
    (Plan.mk probe46 (64)).numWarps = 4 ∧
    (Plan.mk probe46 (64)).numStages = 1 := by decide

def probe47 : Request :=
  ⟨⟨32,0,64,1000,23,2⟩,⟨32,7,67,2000,31,3⟩,
    ⟨32,64,300,4⟩,⟨32,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe47 =
    some ⟨probe47,64⟩ := rfl
example : (Plan.mk probe47 (64)).grid.dims = [0,2] ∧
    (Plan.mk probe47 (64)).numWarps = 4 ∧
    (Plan.mk probe47 (64)).numStages = 1 := by decide

def probe48 : Request :=
  ⟨⟨32,1,64,1000,23,2⟩,⟨32,8,67,2000,31,3⟩,
    ⟨32,64,300,4⟩,⟨32,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe48 =
    some ⟨probe48,64⟩ := rfl
example : (Plan.mk probe48 (64)).grid.dims = [1,2] ∧
    (Plan.mk probe48 (64)).numWarps = 4 ∧
    (Plan.mk probe48 (64)).numStages = 1 := by decide

def probe49 : Request :=
  ⟨⟨32,4,64,1000,23,2⟩,⟨32,11,67,2000,31,3⟩,
    ⟨32,64,300,4⟩,⟨32,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe49 =
    some ⟨probe49,64⟩ := rfl
example : (Plan.mk probe49 (64)).grid.dims = [1,2] ∧
    (Plan.mk probe49 (64)).numWarps = 4 ∧
    (Plan.mk probe49 (64)).numStages = 1 := by decide

def probe50 : Request :=
  ⟨⟨32,5,64,1000,23,2⟩,⟨32,12,67,2000,31,3⟩,
    ⟨32,64,300,4⟩,⟨32,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe50 =
    some ⟨probe50,64⟩ := rfl
example : (Plan.mk probe50 (64)).grid.dims = [2,2] ∧
    (Plan.mk probe50 (64)).numWarps = 4 ∧
    (Plan.mk probe50 (64)).numStages = 1 := by decide

def probe51 : Request :=
  ⟨⟨32,8,64,1000,23,2⟩,⟨32,15,67,2000,31,3⟩,
    ⟨32,64,300,4⟩,⟨32,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe51 =
    some ⟨probe51,64⟩ := rfl
example : (Plan.mk probe51 (64)).grid.dims = [2,2] ∧
    (Plan.mk probe51 (64)).numWarps = 4 ∧
    (Plan.mk probe51 (64)).numStages = 1 := by decide

def probe52 : Request :=
  ⟨⟨33,0,64,1000,23,2⟩,⟨33,7,67,2000,31,3⟩,
    ⟨33,64,300,4⟩,⟨33,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe52 =
    some ⟨probe52,64⟩ := rfl
example : (Plan.mk probe52 (64)).grid.dims = [0,3] ∧
    (Plan.mk probe52 (64)).numWarps = 4 ∧
    (Plan.mk probe52 (64)).numStages = 1 := by decide

def probe53 : Request :=
  ⟨⟨33,1,64,1000,23,2⟩,⟨33,8,67,2000,31,3⟩,
    ⟨33,64,300,4⟩,⟨33,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe53 =
    some ⟨probe53,64⟩ := rfl
example : (Plan.mk probe53 (64)).grid.dims = [1,3] ∧
    (Plan.mk probe53 (64)).numWarps = 4 ∧
    (Plan.mk probe53 (64)).numStages = 1 := by decide

def probe54 : Request :=
  ⟨⟨33,4,64,1000,23,2⟩,⟨33,11,67,2000,31,3⟩,
    ⟨33,64,300,4⟩,⟨33,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe54 =
    some ⟨probe54,64⟩ := rfl
example : (Plan.mk probe54 (64)).grid.dims = [1,3] ∧
    (Plan.mk probe54 (64)).numWarps = 4 ∧
    (Plan.mk probe54 (64)).numStages = 1 := by decide

def probe55 : Request :=
  ⟨⟨33,5,64,1000,23,2⟩,⟨33,12,67,2000,31,3⟩,
    ⟨33,64,300,4⟩,⟨33,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe55 =
    some ⟨probe55,64⟩ := rfl
example : (Plan.mk probe55 (64)).grid.dims = [2,3] ∧
    (Plan.mk probe55 (64)).numWarps = 4 ∧
    (Plan.mk probe55 (64)).numStages = 1 := by decide

def probe56 : Request :=
  ⟨⟨33,8,64,1000,23,2⟩,⟨33,15,67,2000,31,3⟩,
    ⟨33,64,300,4⟩,⟨33,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe56 =
    some ⟨probe56,64⟩ := rfl
example : (Plan.mk probe56 (64)).grid.dims = [2,3] ∧
    (Plan.mk probe56 (64)).numWarps = 4 ∧
    (Plan.mk probe56 (64)).numStages = 1 := by decide

def probe57 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨16,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe57 = none := by decide

def probe58 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,10,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨17,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe58 = none := by decide

def probe59 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨17,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe59 = none := by decide

def probe60 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨16,10,67,2000,31,3⟩,
    ⟨16,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe60 = none := by decide

def probe61 : Request :=
  ⟨⟨17,5,64,1000,23,2⟩,⟨16,10,67,2000,31,3⟩,
    ⟨17,64,300,4⟩,⟨16,65,400,5⟩,1.0⟩
example : prepare (fun _ _ => some (64)) probe61 = none := by decide

end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
