package(
    default_visibility = ["//:__pkg__"],
)

py_library(
    name = "models_audio_vqvae",
    srcs = [
        "models/audio_vqvae.py",
    ],
    srcs_version = "PY3ONLY",
    deps = [
        "@wavenet//:layers",
    ],
)

py_binary(
    name = "train_audio",
    srcs = [
        "train_audio.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":models_audio_vqvae",
    ],
)