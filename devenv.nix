{ pkgs, ... }:
{
  packages = with pkgs; [
    vscode-json-languageserver
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
