import urllib.request
import json
import random
import typer
from src import food_sectors, countries, ner_prompt
from rich.progress import track
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import re

app = typer.Typer()

# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


def extract_json_from_text(text):
    pattern = re.compile(r'<start.*?>\s*(\{.*?\})\s*<end>', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def create_json_prompt_for_synthetic_data(base_prompt, **kwargs):
    """Create a JSON prompt for synthetic data generation."""
    # Use dictionary comprehension to filter out 'n/a' values and to keep the code flexible
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}

    # Create a string of attributes for the <start> tag, excluding any 'n/a' values
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])

    # Adding the dynamically created attributes string to the prompt
    base_prompt += f"""
    <start {attributes_string}>
    """

    return base_prompt


def query_model(content, model="llama3:instruct", url="http://localhost:11434/api/chat", role="user", **kwargs):
    """Query the model with the given content and return the response."""
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "temperature": 1.,  # for deterministic responses
        "top_p": 1,
        "messages": [
            {"role": role, "content": content}
        ]
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def generate_from_prompts(prompts):
    """Generate synthetic data from a list of prompts."""
    all_outs = []
    typer.secho("Generating synthetic data...", fg=typer.colors.MAGENTA)
    with progress_bar as p:
        for value in p.track(prompts):
            # for value in track(prompts, description="[green]Generating"):
            try:
                result = query_model(value)
                json_string = extract_json_from_text(result)
                if json_string:
                    js = json.loads(extract_json_from_text(result))
                    all_outs.append(js)
                    with open('data/raw_data.jsonl', 'a', encoding="utf-8") as file:
                        json_string = json.dumps(js, ensure_ascii=False)
                        file.write(json_string + '\n')
                else:
                    continue
            except Exception as e:
                typer.secho(f"Error: {e}", fg=typer.colors.RED)
                continue
    typer.secho("Synthetic data generated successfully!", fg=typer.colors.GREEN)


@app.command()
def main(samples: int, language: str = "english", types: str = "meals description"):
    """Main function to generate synthetic data."""
    typer.secho(f"Generating {samples} samples...", fg=typer.colors.MAGENTA)
    all_prompts = []
    for i in range(samples):
        prompt = create_json_prompt_for_synthetic_data(
            base_prompt=ner_prompt,
            language=language,
            types_of_text=types,
            sector=random.choice(food_sectors),
            country=random.choice(countries)
        )
        all_prompts.append(prompt)

    typer.secho("Prompts generated successfully!", fg=typer.colors.GREEN)

    generate_from_prompts(all_prompts)


if __name__ == "__main__":
    app()
