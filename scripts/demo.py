"""Interactive demo of LLSD steering.

Usage:
    python scripts/demo.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from llsd import SteeringModel

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Interactive LLSD demo")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--vectors",
        type=str,
        default="vectors/layer_16_mean_diff.pt",
        help="Path to steering vectors",
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")

    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]LLSD: LLM on LSD[/bold cyan]\n"
            "Controlled Divergent Thinking via Activation Steering",
            border_style="cyan",
        )
    )

    # Load model
    console.print(f"\n[yellow]Loading model: {args.model}[/yellow]")
    model = SteeringModel.from_pretrained(args.model, load_in_8bit=args.load_in_8bit)

    # Load vectors
    console.print(f"[yellow]Loading steering vectors: {args.vectors}[/yellow]")
    model.load_vectors(args.vectors)

    # Interactive loop
    console.print("\n[green]Model loaded! Enter prompts to see steered outputs.[/green]")
    console.print("[dim]Type 'quit' to exit, 'alpha X' to set alpha value[/dim]\n")

    current_alpha = 1.5

    while True:
        prompt = Prompt.ask("\n[bold blue]Prompt[/bold blue]")

        if prompt.lower() == "quit":
            break

        if prompt.lower().startswith("alpha "):
            try:
                current_alpha = float(prompt.split()[1])
                console.print(f"[green]Set alpha to {current_alpha}[/green]")
            except (IndexError, ValueError):
                console.print("[red]Invalid alpha value. Use: alpha 1.5[/red]")
            continue

        # Generate with different alphas
        for alpha in [0.0, current_alpha]:
            model.set_divergence(alpha)
            output = model.generate(prompt, max_new_tokens=150)

            style = "Normal" if alpha == 0.0 else f"Steered (α={alpha})"
            console.print(
                Panel(
                    output,
                    title=f"[bold]{style}[/bold]",
                    border_style="green" if alpha > 0 else "dim",
                )
            )


if __name__ == "__main__":
    main()
