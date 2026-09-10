"""Import-only FA4 environment check; does not require or execute a GPU."""

from importlib.metadata import version


def main() -> None:
    print(f"flash-attn-4: {version('flash-attn-4')}")
    print(f"nvidia-cutlass-dsl: {version('nvidia-cutlass-dsl')}")
    from flash_attn.cute import flash_attn_func

    print(f"flash_attn_func import: ok ({flash_attn_func.__module__})")


if __name__ == "__main__":
    main()
