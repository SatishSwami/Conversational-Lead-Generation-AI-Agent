"""
AutoStream Agent — CLI Entrypoint

Run with:
    python main.py

Optional overrides:
    python main.py --provider google
    python main.py --provider google --model gemini-3.8-flash
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    """Parse optional CLI arguments."""
    parser = argparse.ArgumentParser(
        description="AutoStream Conversational AI Agent"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "google"],
        help=(
            "LLM provider to use. "
            "Defaults to LLM_PROVIDER from environment."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Specific model name. "
            "Defaults to LLM_MODEL from environment."
        ),
    )

    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the welcome banner.",
    )

    return parser.parse_args()


def print_banner():
    """Print the CLI welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          AutoStream AI Sales Assistant — Powered by Inflx   ║
║          Type 'quit' or 'exit' to end the session           ║
║          Type 'reset' to start a new conversation           ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def resolve_configuration(args):
    """
    Resolve provider and model.

    Priority:
        CLI argument > environment configuration > application defaults.
    """
    from core.config import get_settings

    settings = get_settings()

    provider = args.provider or settings.llm_provider
    model = args.model or settings.llm_model

    return provider, model


def validate_env(provider: str):
    """Validate that the selected LLM provider has an API key."""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    if provider not in env_map:
        print(f"\n[ERROR] Unsupported LLM provider: {provider}")
        print(
            "Supported providers: anthropic, openai, google\n"
        )
        sys.exit(1)

    key_name = env_map[provider]

    if not os.environ.get(key_name):
        print(f"\n[ERROR] Missing API key: {key_name}")
        print(
            "Set it in your .env file or as an environment variable.\n"
        )
        sys.exit(1)


def run_cli():
    """Start the interactive AutoStream CLI."""
    args = parse_args()

    provider, model = resolve_configuration(args)

    validate_env(provider)

    if not args.no_banner:
        print_banner()

    print(
        f"[CONFIG] Provider: {provider} | "
        f"Model: {model or 'provider default'}"
    )
    print(
        "[CONFIG] Knowledge base: ./knowledge_base/autostream_kb.json\n"
    )

    # Import after environment/configuration is loaded.
    from agent.graph import AutoStreamAgent

    agent = AutoStreamAgent(
        provider=provider,
        model=model,
    )

    print(
        "Aria: Hello! Welcome to AutoStream. "
        "I'm Aria, your AI assistant. "
        "How can I help you today?\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print(
                "\n\nAria: Thanks for chatting! "
                "Have a great day. 👋"
            )
            break

        if not user_input:
            continue

        if user_input.lower() in (
            "quit",
            "exit",
            "bye",
            "goodbye",
        ):
            print(
                "\nAria: Thanks for your interest in AutoStream! "
                "Feel free to reach out anytime. "
                "Goodbye! 👋"
            )
            break

        if user_input.lower() == "reset":
            agent.reset()

            print(
                "\n[Session reset. Starting fresh conversation.]\n"
            )

            print(
                "Aria: Hello again! "
                "How can I help you with AutoStream today?\n"
            )
            continue

        if user_input.lower() == "debug":
            print(f"\n[DEBUG] Turn: {agent.turn_count}")
            print(
                f"[DEBUG] Intent: "
                f"{agent._state.get('current_intent')}"
            )
            print(
                f"[DEBUG] Lead active: "
                f"{agent._state.get('lead_collection_active')}"
            )
            print(
                f"[DEBUG] Lead captured: "
                f"{agent._state.get('lead_captured')}"
            )
            print(
                f"[DEBUG] Collected: "
                f"{agent._state.get('lead_collector_state', {}).get('collected', {})}"
            )
            print()
            continue

        try:
            response = agent.chat(user_input)

            print(f"\nAria: {response}\n")

            if agent.is_lead_captured:
                print(
                    "\n[✓ Lead successfully captured. "
                    "Session will continue for any follow-up questions.]\n"
                )

        except Exception as exc:
            print(
                f"\n[ERROR] Agent encountered an issue: {exc}"
            )
            print(
                "Please try again or type 'reset' to restart.\n"
            )


if __name__ == "__main__":
    run_cli()