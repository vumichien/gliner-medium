import json
from src import (
    extract_entities,
    save_data_to_file,
    analyze_generic_data,
    upload_to_hf,
    download_from_hf,
    extract_entities_japanese,
    top_common_meal_ner,
    has_top_common_ner,
)
import typer

app = typer.Typer()


@app.command()
def parse_data(input_file: str, output_file: str = "", analyze: bool = False, jp: bool = False):
    """Main function to parse data and extract named entities."""
    typer.secho(f"Reading data from {input_file}...", fg=typer.colors.MAGENTA)
    if not jp:
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        parsed_data = extract_entities(data)
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        parsed_data = extract_entities_japanese(data)
    typer.secho("Data read successfully!", fg=typer.colors.GREEN)
    if len(output_file) > 0:
        typer.secho(f"Saving data to {output_file}...", fg=typer.colors.MAGENTA)
        save_data_to_file(parsed_data, output_file)
        typer.secho("Data saved successfully!", fg=typer.colors.GREEN)

    if analyze:
        typer.secho("Analyzing data...", fg=typer.colors.MAGENTA)
        analyze_generic_data(parsed_data)
        typer.secho("Data analysis complete!", fg=typer.colors.GREEN)


@app.command()
def filter_data(input_file: str, output_file: str):
    typer.secho(f"Reading data from {input_file}...", fg=typer.colors.MAGENTA)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    typer.secho(f"Length of data: {len(data)}", fg=typer.colors.GREEN)
    filtered_data = [item for item in data if has_top_common_ner(item["ner"], top_common_meal_ner)]
    typer.secho(f"Filtered data contains {len(filtered_data)} examples.", fg=typer.colors.GREEN)
    typer.secho(f"Saving filtered data to {output_file}...", fg=typer.colors.MAGENTA)
    save_data_to_file(filtered_data, output_file)
    typer.secho("Filtered data saved successfully!", fg=typer.colors.GREEN)


@app.command()
def upload_data(repo_name: str, file_path: str):
    typer.secho(f"Uploading data to {repo_name}...", fg=typer.colors.MAGENTA)
    upload_to_hf(repo_name, file_path)
    typer.secho("Data uploaded successfully!", fg=typer.colors.GREEN)


@app.command()
def download_data(repo_name: str, file_path: str):
    typer.secho(f"Downloading data from {repo_name}...", fg=typer.colors.MAGENTA)
    download_from_hf(repo_name, file_path)
    typer.secho("Data downloaded successfully!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
