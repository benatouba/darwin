{ pkgs, lib, ... }:
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

  # Commit-time checks, installed into .git/hooks by `devenv shell` and run by
  # prek on staged files, plus the full test suite. `ruff check` is not a hook
  # yet: pyproject selects every rule and the tree carries a large existing
  # baseline, so it would refuse almost every commit until that is paid down.
  git-hooks.package = pkgs.prek;
  git-hooks.hooks = {
    check-toml.enable = true;
    check-yaml.enable = true;
    end-of-file-fixer.enable = true;
    trim-trailing-whitespace.enable = true;
    local-ruff-format = {
      enable = true;
      name = "ruff format";
      entry = "ruff format";
      language = "system";
      types = [ "python" ];
    };
    local-pytest = {
      enable = true;
      name = "pytest";
      entry = "uv run --frozen --no-sync python -m pytest -q";
      language = "system";
      pass_filenames = false;
      types = [ "python" ];
    };
  };

  # `devenv test` would otherwise run `prek run -a` over every file, which the
  # unformatted legacy files fail. The installed pre-commit hook still runs on
  # staged files at commit time.
  tasks."devenv:git-hooks:run".before = lib.mkForce [ ];
}
