from src.core.finai import FinAI
import sys


def main():
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print("Usage: python run_prompt.py \"your prompt here\"")
        return 1

    app = FinAI()
    if not app.initialize():
        print("Model not found. Train first: python main.py train <path-to-text>")
        return 1

    out = app.generate_response(prompt)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
