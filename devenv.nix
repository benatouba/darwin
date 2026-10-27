{ pkgs, ... }:
{
  packages = with pkgs; [
    vscode-json-languageserver
    # Explicit native-diagnostic builds and tests (darwin.native_diagnostics).
    gfortran
    glibc.bin
  ];

  languages = {
    python = {
      enable = true;
      package = pkgs.python314.withPackages (
        ps: with ps; [
          pytest
          ruff
        ]
      );
      lsp.package = pkgs.ty;
      venv.enable = true;
      uv = {
        enable = true;
        sync.enable = true;
      };
    };
  };

  enterShell = ''
    if [ -f .venv/bin/activate ]; then
      source .venv/bin/activate
    fi
  '';
}
