"""All ad-hoc CLI jobs run through observable Prefect flows."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="BankingEnv document-grounded v3")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("regenerate")
    build.add_argument("--out", required=True)
    build.add_argument("--development-seed", default="bankingenv-v3-development-20261020")
    build.add_argument("--variants", type=int, default=3)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--policy", choices=("reference", "live"), default="reference")
    run.add_argument("--turn-limit", type=int, default=24)
    run.add_argument("--limit", type=int)
    grade = sub.add_parser("grade")
    grade.add_argument("--dataset", required=True)
    grade.add_argument("--transcripts", required=True)
    grade.add_argument("--out", required=True)
    args = parser.parse_args()
    from .flow import evaluate, regenerate, grade_transcripts
    if args.command == "regenerate":
        regenerate(args.out, args.development_seed, args.variants)
    elif args.command == "grade":
        grade_transcripts(args.dataset, args.transcripts, args.out)
    else:
        report = evaluate(args.dataset, args.out, args.policy, args.turn_limit, args.limit)
        if report["execution_errors"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
