# Saved research preset: adaptive top-2/3 prefill

The separate expert-routing research build is configured with
`STRATA_RESEARCH_ADAPTIVE_PREFILL=ON`. `build_xb.sh` preserves this setting on rebuild.
The ordinary CMake default is OFF; this does not change the deployed Strata checkout.

The saved policy is:

- Use two routed experts when their cumulative weight is at least 0.35 of the
  original selected top-ten weight; otherwise use three. Renormalize retained weights.
- Never select one expert in this preset. The shared expert stays enabled.
- Apply only to prefill chunks of at least 64 tokens. Short chunks and the decode
  path keep all ten routed experts. `STRATA_XB_MIN_T` can increase that minimum.

Run `build/strata` with the normal model/server arguments; no temporary policy file
is required. The first qualifying prefill logs the active preset.

## Overrides and rollback

`STRATA_XB_ADAPT_MASS=0` disables the preset at runtime. For example:

```sh
STRATA_XB_ADAPT_MASS=0 ./build/strata --help
```

An explicit `STRATA_XB_ADAPT_FILE` takes precedence, including a zero `full_from`
value for native reference runs. Explicit `STRATA_XB_FILE` or `STRATA_XB_K` settings
also suppress the saved preset. Existing experiment harnesses retain their controls.
Setting the mass variable does not supersede these explicit experiment settings.

To disable the default at build time, configure with
`-DSTRATA_RESEARCH_ADAPTIVE_PREFILL=OFF` and rebuild. `build_xb.sh` explicitly opts
back into the research preset, so change its option too if retaining that rollback.

## Measured scope

On a P100 16 GB, Flash-Next IQ3_S, the file-controlled 0.35 policy achieved
100% teacher-forced next-token agreement over 488 scored positions and mean
KL(native || candidate) 0.01812 across 12 unseen tasks at 1k/2k/4k/8k.
Its aggregate targets passed at every tested length. This is not an exactness
guarantee for every prompt or for long free-running generation.

Eight paired 8k timing comparisons measured 18.174s for fixed top-3 versus
17.905s adaptive, a 1.48% additional reduction. The original fixed top-3 reduction
relative to native top-ten prefill accounts for most of the total gain.

Those results were collected using explicit per-request policy files. The saved
chunk-based preset uses the same selection and pruning loop. Its wiring has a
separate default-mode smoke check. Chunks below 64 tokens stay native in the preset.
Other GPU backends, production concurrency, and contexts beyond 8k have not been
validated for this policy.

## Default-mode verification

The rebuilt research binary was launched without policy-file, fixed-K, mass, or
minimum-chunk environment overrides. An 8k code calibration prompt activated the
saved preset automatically: 383,664 token-layer decisions, all selecting two or
three experts, zero selecting one. Teacher-forced agreement was 100%, mean KL
0.015025. A 32-token input produced no adaptive-routing events; generated
continuations likewise produced no additional adaptive-prefill decisions. The
model process was closed after this check. CUDA 12.9 / sm_60 build succeeded.

The previously validated binary and pre-change source files are preserved in
`work/research-goal/research-preset/` in the Codex research workspace. The original
frozen experiment manifests retain their original hashes; no old measurements
were relabeled as measurements of this rebuilt binary.
