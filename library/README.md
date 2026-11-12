# Library-only dependency view

Downstream packages that own copies of `bench/examples` can select this Git
subdirectory in their Lake dependency:

```toml
[[require]]
name = "veritile"
git = "https://github.com/VeriInfra-Lab/VeriTile.git"
rev = "<pinned-commit>"
subDir = "library"
```

This view uses the same library sources, module names, toolchain, Mathlib
lockfile and build directory as the root package. It omits only the
`TritonBenchSpecExamples` target from module discovery, allowing the downstream
package to provide its own worked examples without same-name ambiguity.
The root package and its CI examples retain their existing configuration.

`lakefile.toml` is a generated view of `../lakefile.toml`: copy the root
configuration, remove `TritonBenchSpecExamples`, and set `srcDir`, `buildDir`
and `manifestFile` to the parent paths shown in this file. Run
`python3 -m unittest scripts.test_library_dependency_profile` after changing
the root configuration; the regression requires the exact intended difference.
