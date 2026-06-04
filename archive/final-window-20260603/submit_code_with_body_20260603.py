import argparse
import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--version", type=int)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    try:
        response = api.competition_submit_code(
            args.file_name,
            args.message,
            args.competition,
            args.kernel,
            args.version,
            quiet=True,
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "exception": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": str(exc),
        }
        response = getattr(exc, "response", None)
        if response is not None:
            payload.update(
                {
                    "status_code": getattr(response, "status_code", None),
                    "reason": getattr(response, "reason", None),
                    "body": getattr(response, "text", ""),
                }
            )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 1

    payload = {
        "ok": True,
        "message": getattr(response, "message", ""),
        "raw": str(response),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
