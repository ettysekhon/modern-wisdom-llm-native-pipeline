from .cli import build_parser
from .schema import write_schemas


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Write schema files idempotently so /data/contracts/schemas isn’t empty
    write_schemas()
    args.func(args)


if __name__ == "__main__":
    main()
