import httpx
import asyncio
import time
import statistics
from collections import Counter
from rich.console import Console
from rich.table import Table

# --- SCRIPT CONFIGURATION ---

# !!! REPLACE THIS with your ALB's DNS Name !!!
# This now points to the root of your public-facing application.
ALB_DNS_NAME = "http://rag-app-alb-1234567890.us-east-1.elb.amazonaws.com"
ENDPOINT_URL = ALB_DNS_NAME  # We are hitting the main frontend page

# --- LOAD TEST PARAMETERS (can remain the same) ---
START_CONCURRENCY = 10
STEP_CONCURRENCY = 10
STAGE_DURATION_SECONDS = 60
REQUEST_TIMEOUT = 30.0

# --- SCRIPT LOGIC ---

console = Console()

async def send_request(client: httpx.AsyncClient, stage_end_time: float):
    """Sends a single GET request to the frontend."""
    if time.time() > stage_end_time:
        return None

    start_time = time.time()
    try:
        # CHANGE: Use .get() and remove the json payload.
        # We are simply requesting the main Streamlit page, not sending data.
        response = await client.get(ENDPOINT_URL, timeout=REQUEST_TIMEOUT)
        latency = time.time() - start_time

        # A 200 OK from the frontend means it successfully served the page.
        if 200 <= response.status_code < 300:
            return ("success", latency, response.status_code)
        else:
            return ("failure", latency, response.status_code)

    except httpx.TimeoutException:
        latency = time.time() - start_time
        return ("failure", latency, "timeout")
    except httpx.RequestError as e:
        latency = time.time() - start_time
        return ("failure", latency, str(type(e).__name__))

# --- The rest of the script (run_stage, process_results, main) can remain exactly the same ---
async def run_stage(concurrency: int):
    """Runs a single load test stage with a given concurrency."""
    console.print(f"\n[bold cyan]Running stage with {concurrency} concurrent requests for {STAGE_DURATION_SECONDS} seconds...[/bold cyan]")

    all_results = []
    stage_end_time = time.time() + STAGE_DURATION_SECONDS

    async with httpx.AsyncClient() as client:
        while time.time() < stage_end_time:
            tasks = [send_request(client, stage_end_time) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks)
            all_results.extend([r for r in results if r is not None])
            await asyncio.sleep(0.01)

    return all_results

def process_results(results, concurrency):
    """Analyzes and prints the results of a stage."""
    successes = [r for r in results if r[0] == "success"]
    failures = [r for r in results if r[0] == "failure"]

    latencies = [r[1] for r in successes]
    failure_reasons = Counter([r[2] for r in failures])

    total_requests = len(results)
    success_rate = (len(successes) / total_requests * 100) if total_requests > 0 else 0
    failure_rate = (len(failures) / total_requests * 100) if total_requests > 0 else 0

    table = Table(title=f"Stage Summary: {concurrency} Concurrent Users")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Requests", str(total_requests))
    table.add_row("Success Rate", f"[green]{success_rate:.2f}%[/green]")
    table.add_row("Failure Rate", f"[red]{failure_rate:.2f}%[/red]")

    if latencies:
        table.add_row("Avg. Latency", f"{statistics.mean(latencies):.2f}s")
        table.add_row("Median Latency (p50)", f"{statistics.median(latencies):.2f}s")
        table.add_row("95th Percentile Latency", f"{statistics.quantiles(latencies, n=100)[94]:.2f}s")

    if failures:
        table.add_row("[bold red]-- Failure Breakdown --[/bold red]", "")
        for reason, count in failure_reasons.items():
            table.add_row(f"  {reason}", str(count))

    console.print(table)
    return failure_rate

async def main():
    """The main function to orchestrate the load test."""
    concurrency = START_CONCURRENCY
    while True:
        results = await run_stage(concurrency)
        if not results:
            console.print("[yellow]No requests were completed in the last stage. Stopping test.[/yellow]")
            break
        
        failure_rate = process_results(results, concurrency)

        if failure_rate > 5.0: 
            console.print(f"\n[bold red]BREAKING POINT DETECTED![/bold red]")
            console.print(f"System became unstable at ~{concurrency} concurrent requests.")
            break

        concurrency += STEP_CONCURRENCY
        console.print(f"Waiting 10 seconds before starting the next stage...")
        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Load test stopped by user.[/yellow]")
