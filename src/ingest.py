import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_manager import GeminiManager
from utils import ConfigValidator

console = Console()


class PDFIngestor:
    """Class responsible for reading the PDF, splitting the text, and storing it in the vector database."""

    def __init__(self, pdf_path: str, db_url: str, collection_name: str, gemini_manager: GeminiManager):
        self.pdf_path = pdf_path
        self.db_url = db_url
        self.collection_name = collection_name
        self.embeddings = gemini_manager.get_embeddings()

    def run(self):
        """Main method that executes the ingestion pipeline with visual progress."""
        console.print(
            Panel.fit(
                "🚀 [bold blue]Starting Ingestion Pipeline[/bold blue]",
                border_style="blue"
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            task1 = progress.add_task("[cyan]Reading and splitting PDF...", total=1)

            docs = PyPDFLoader(str(self.pdf_path)).load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                add_start_index=False
            )

            splits = splitter.split_documents(docs)

            if not splits:
                raise RuntimeError("No content found in the PDF.")

            progress.update(task1, advance=1)

            task2 = progress.add_task("[magenta]Enriching documents...", total=len(splits))

            enriched_docs = []
            for d in splits:
                enriched_docs.append(
                    Document(
                        page_content=d.page_content,
                        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
                    )
                )
                progress.update(task2, advance=1)

            task3 = progress.add_task("[green]Saving to PostgreSQL (Generating Embeddings)...", total=1)

            ids = [f"doc-{i}" for i in range(len(enriched_docs))]

            store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.db_url,
                use_jsonb=True,
            )

            store.add_documents(documents=enriched_docs, ids=ids)
            progress.update(task3, advance=1)

        console.print("\n[bold green]✅ Ingestion completed successfully![/bold green]")
        console.print(f"📊 Total chunks processed: [bold]{len(enriched_docs)}[/bold]\n")


if __name__ == "__main__":
    try:
        config = ConfigValidator.load_and_validate()

        gemini_manager = GeminiManager(
            embedding_model_name=config["GOOGLE_EMBEDDING_MODEL"]
        )

        ingestor = PDFIngestor(
            pdf_path=config["PDF_PATH"],
            db_url=config["DATABASE_URL"],
            collection_name=config["PG_VECTOR_COLLECTION_NAME"],
            gemini_manager=gemini_manager
        )

        ingestor.run()

    except Exception as e:
        console.print(f"[bold red]❌ Fatal error during ingestion:[/bold red] {e}")