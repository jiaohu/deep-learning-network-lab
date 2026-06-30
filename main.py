import torch


def main() -> None:
    is_mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {is_mps_available}")


if __name__ == "__main__":
    main()
