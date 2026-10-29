import VeriTile.Triton.DSL

/-! Port declarations retain the complete ordered interface without emitting
instructions, changing operand precision, or asserting memory safety. -/
namespace KernelPortMetadataTests
open VeriTile.Triton

def inferred (X O : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, O = tl.float32
  p = X + $(1)
  value = tl.load(p)
  tl.store(O, value)
}

def declared (X O Unused : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[Unused, X, X], outputs=[O, Unused]
  tl.region X = tl.float32, O = tl.float32
  p = X + $(1)
  value = tl.load(p)
  tl.store(O, value)
}

example (X O Unused : RegionName) :
    declared X O Unused =
      .mk [Unused, X, X] [O, Unused] (inferred X O).surfaceBody := rfl

def empty (Unused : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[], outputs=[Unused]
}

example (Unused : RegionName) : empty Unused = .mk [] [Unused] [] := rfl

def typed (X : Region .fp32) (O : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X.cast], outputs=[O]
  value = tl.load(X)
  tl.store(O, value)
}

def typedInferred (X : Region .fp32) (O : RegionName) : ComputeKernel := triton {
  value = tl.load(X)
  tl.store(O, value)
}

example (X : Region .fp32) (O : RegionName) :
    typed X O = .mk [X.cast] [O] (typedInferred X O).surfaceBody := rfl

-- Typed terms in metadata must not alter the legacy pointer-cast policy.
def legacy (X O : RegionName) : ComputeKernel := triton {
  value = tl.load(X)
  tl.store(O, value)
}

def annotated (X O : RegionName) (unused : Region .fp32) : ComputeKernel := triton {
  tl.ports inputs=[X, (unused : Region .fp32).cast], outputs=[O]
  value = tl.load(X)
  tl.store(O, value)
}

example (X O : RegionName) (unused : Region .fp32) :
    annotated X O unused =
      .mk [X, unused.cast] [O] (legacy X O).surfaceBody := rfl

/-- error: triton: only one top-level tl.ports declaration is allowed -/
#guard_msgs in
def duplicate : ComputeKernel := triton {
  tl.ports inputs=[], outputs=[]
  tl.ports inputs=[], outputs=[]
}

end KernelPortMetadataTests
