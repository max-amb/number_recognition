{
  description = "Flake for rust dev";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, rust-overlay, ... }:
  let
    system = "x86_64-linux";
    overlays = [ (import rust-overlay) ];
    pkgs = import nixpkgs {
      inherit system overlays;
    };
  in {
    devShells."x86_64-linux".default = with pkgs; mkShell {
      buildInputs = [
        pkg-config
        (rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        })
        gdb
        python313
        python313Packages.matplotlib
        python313Packages.torch
        python313Packages.torchvision
      ];
      shellHook = ''
        export SHELL=/run/current-system/sw/bin/bash
      '';
    };

  };
}

