"""Shared command-line configuration for the SpanDC pipeline."""

DEFAULT_MINIMUM_INTERVAL_LENGTH = 20
DEFAULT_MAXIMUM_PREDICATE_LENGTH = 3
DEFAULT_MAX_SPANS = 32


def add_core_arguments(parser) -> None:
    parser.add_argument(
        "--minimum-interval-length", "--min-interval-length",
        dest="minimum_interval_length", type=int,
        default=DEFAULT_MINIMUM_INTERVAL_LENGTH,
        help="Minimum interval/span length (default: 20).",
    )
    parser.add_argument(
        "--maximum-predicate-length", "--max-predicate-length",
        dest="maximum_predicate_length", type=int,
        default=DEFAULT_MAXIMUM_PREDICATE_LENGTH,
        help="Maximum predicates per rule (default: 3).",
    )
    parser.add_argument(
        "--max-spans", "--k-max",
        dest="max_spans", type=int, default=DEFAULT_MAX_SPANS,
        help="Maximum number of summary spans, K_max (default: 32).",
    )


def validate_core_arguments(args) -> None:
    if args.minimum_interval_length < 2:
        raise ValueError("minimum interval length must be at least 2")
    if args.maximum_predicate_length < 1:
        raise ValueError("maximum predicate length must be at least 1")
    if args.max_spans < 1:
        raise ValueError("K_max must be at least 1")


def validate_artifact_config(metadata: dict, args) -> None:
    validate_core_arguments(args)
    expected = {
        "minimum_interval_length": args.minimum_interval_length,
        "maximum_predicate_length": args.maximum_predicate_length,
        "max_spans": args.max_spans,
    }
    mismatches = [
        f"{name}: artifact={metadata.get(name)!r}, requested={value!r}"
        for name, value in expected.items() if metadata.get(name) != value
    ]
    if mismatches:
        raise ValueError("core configuration does not match the mined artifact: " + "; ".join(mismatches))
