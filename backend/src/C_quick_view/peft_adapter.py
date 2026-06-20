from pathlib import Path


def find_adapter_files(adapter_directory: str | Path) -> list[Path]:
    adapter_directory = Path(adapter_directory)

    if not adapter_directory.exists():
        return []

    valid_extensions = {
        ".safetensors",
        ".bin",
        ".json",
        ".pt",
        ".pth",
    }

    return [
        file
        for file in adapter_directory.rglob("*")
        if file.is_file() and file.suffix.lower() in valid_extensions
    ]


def print_adapter_summary(adapter_directory: str | Path, adapter_name: str) -> None:
    files = find_adapter_files(adapter_directory)

    print(f"\nAdaptador {adapter_name}:")
    print(f"Ruta: {adapter_directory}")

    if not files:
        print("Estado: no se encontraron archivos de adaptador.")
        return

    print("Archivos encontrados:")

    for file in files:
        print(f"- {file}")