{
  pkgs,
  lib,
  ...
}:
let
  buildInputs = with pkgs; [
    stdenv.cc.cc
    libuv
    zlib
  ];
in
{
  packages = with pkgs; [
    python312
  ];

  env = {
    UV_PYTHON = "${pkgs.python312}/bin/python";
    LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
  };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  enterShell = ''
    uv sync
    . .devenv/state/venv/bin/activate
  '';

  enterTest = ''
    uv sync
    . .devenv/state/venv/bin/activate
    pytest
  '';
}
