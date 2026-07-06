# pdhg

Small C++ 23/Metal project for learning and building a PDHG implementation.

## Build

Build:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run naming checks with clang-tidy:

```sh
cmake --build build --target lint
```

The `lint` target requires `clang-tidy` on `PATH`. The naming policy lives in
`.clang-tidy`.

Format C++ sources with clang-format:

```sh
cmake --build build --target format
cmake --build build --target format-check
```

The formatting targets require `clang-format` on `PATH`. The formatting policy
lives in `.clang-format`.

Or from the parent repo:

```sh
cmake -S pdhg -B pdhg/build
cmake --build pdhg/build
ctest --test-dir pdhg/build --output-on-failure
```

The project assumes an Apple Silicon Mac. CMake fetches the pinned Apple
`metal-cpp`, `magic_enum`, and Microsoft GSL headers and links Foundation,
QuartzCore, and Metal. Header-only third-party dependencies should be added
through pinned CMake `FetchContent` entries unless there is a specific reason
to vendor the files into the repo.

Keep the `NS_PRIVATE_IMPLEMENTATION`, `CA_PRIVATE_IMPLEMENTATION`, and
`MTL_PRIVATE_IMPLEMENTATION` defines in exactly one translation unit.


## Code layout

Since this is mostly about learning, I will begin by defining some useful linear algebra objects.
