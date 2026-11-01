"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    usage_metadata = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage_metadata, "prompt_token_count", None)
    output_tokens = getattr(usage_metadata, "candidates_token_count", None)
    print(f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}")
    print(f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response.text


def interactive() -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    print("会話を開始します。終了するには 'exit' または 'quit' を入力してください。")

    while True:
        try:
            question = input("[質問] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会話を終了します。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("会話を終了します。")
            break

        try:
            response = chat.send_message(question)
            print_usage(response)
            print(response.text)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.question is not None:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
        else:
            interactive()
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
