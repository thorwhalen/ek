# ek-rs

Optional native (Rust) kernels for [ek](https://github.com/thorwhalen/ek). Install
with `pip install ek[rust]`. Nothing imports `ek_rs` directly: ek selects it through
its private backend seams, and the default is always pure Python. The Rust source
lives in `rust/` of the ek repository, and `misc/docs/port-log.md` there records how
it was ported and tested.
