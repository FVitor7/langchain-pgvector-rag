import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from llm_manager import GeminiManager
from search import QASystem
from utils import ConfigValidator

console = Console()

MSG_NOT_FOUND = "I do not have enough information to answer your question."

def main():
    try:
        with console.status("[bold green]Loading settings and models...", spinner="dots"):
            config = ConfigValidator.load_and_validate()
            
            gemini_manager = GeminiManager(
                embedding_model_name=config.get("GOOGLE_EMBEDDING_MODEL")
            )
            
            qa_system = QASystem(
                db_url=config.get("DATABASE_URL"),
                collection_name=config.get("PG_VECTOR_COLLECTION_NAME"),
                gemini_manager=gemini_manager
            )

        console.print(Panel(
            "[bold cyan]RAG-Powered PDF Assistant[/bold cyan]\n"
            "[white]Semantic search powered by LangChain, pgVector and Gemini.[/white]\n\n"
            "[dim]Ask questions about the document. Type 'exit' to quit.[/dim]",
            border_style="blue",
            title="🤖 AI Document Chat",
            subtitle="Full Cycle MBA - Software Engineering with AI"
        ))

        while True:
            question = Prompt.ask("\n[bold yellow]❓ QUESTION[/bold yellow]")

            if question.lower() in ["exit", "quit"]:
                console.print("\n[bold red]Closing chat. See you soon![/bold red] 👋")
                break
            
            if not question.strip():
                continue

            with console.status("[italic cyan]Searching database and generating answer...", spinner="arc"):
                answer = qa_system.search_prompt(question, k=10)

            console.print("\n[bold green]✨ ANSWER:[/bold green]")

            if MSG_NOT_FOUND in answer:
                console.print(Panel(
                    Markdown(f"⚠️  {answer}"), 
                    border_style="yellow",
                    padding=(1, 2),
                    title="[bold yellow]Warning: Out of Context[/bold yellow]"
                ))
            else:
                console.print(Panel(
                    Markdown(answer), 
                    border_style="green",
                    padding=(1, 2),
                    title="[bold green]Result Found[/bold green]"
                ))
            

            console.print("[dim]" + "─" * console.width + "[/dim]")

    except KeyboardInterrupt:
        console.print("\n\n[bold red]Interrupted by user. Exiting...[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(Panel(
            f"[bold red]Critical execution error:[/bold red]\n{str(e)}", 
            title="[bold red]ERROR[/bold red]", 
            border_style="red"
        ))

if __name__ == "__main__":
    main()