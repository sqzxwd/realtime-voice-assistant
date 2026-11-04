"""Command Line Interface and Rich Terminal UI for Realtime Voice Assistant."""

import asyncio
import logging
import signal
import sys
from typing import List, Optional
import rich
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import sounddevice as sd
import typer

from src.audio.capture import AudioCapture, MicrophoneError, MicrophonePermissionError
from src.audio.player import AudioPlayer, PlayerError
from src.audio.utils import render_vu_bar
from src.client import (
    AudioReceived,
    GeminiLiveProtocolError,
    Interrupted,
    TranscriptReceived,
    TurnComplete,
    VoiceActivity,
    create_client,
)
from src.config import AssistantConfig

# Set up logging to quiet external libraries in terminal
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("realtime-voice-assistant")

app = typer.Typer(
    name="realtime-voice-assistant",
    help="Production-ready real-time two-way voice streaming assistant with Gemini Live API",
    add_completion=False,
)
console = Console()


class AssistantUIState:
    """Stores the dynamic state for the terminal UI."""

    def __init__(self, mode_str: str) -> None:
        self.mode_str = mode_str
        self.status = "LISTENING"  # LISTENING, STREAMING, SPEAKING, BARGE-IN
        self.mic_db: float = -60.0
        self.speaker_db: float = -60.0
        self.transcripts: List[tuple[str, str]] = []  # (role, text)
        self.barge_in_count = 0
        self.is_running = True

    def add_transcript(self, role: str, text: str) -> None:
        if self.transcripts and self.transcripts[-1][0] == role:
            # Append if previous was same speaker
            prev_role, prev_text = self.transcripts[-1]
            self.transcripts[-1] = (prev_role, prev_text + " " + text)
        else:
            self.transcripts.append((role, text))
            if len(self.transcripts) > 6:
                self.transcripts.pop(0)


def create_ui_renderable(state: AssistantUIState) -> Panel:
    """Generate the Rich UI widget for the terminal live display."""
    # 1. Status Badge
    status_text = Text()
    if state.status == "LISTENING":
        status_text.append("● LISTENING ", style="bold green")
        status_desc = "(Listening for your voice...)"
    elif state.status == "STREAMING":
        status_text.append("▲ STREAMING ", style="bold yellow")
        status_desc = "(Sending audio to assistant...)"
    elif state.status == "SPEAKING":
        status_text.append("▼ SPEAKING  ", style="bold cyan")
        status_desc = "(Assistant speaking...)"
    elif state.status == "BARGE-IN":
        status_text.append("⚡ BARGE-IN  ", style="bold magenta")
        status_desc = "(Interruption detected! Audio flushed)"
    else:
        status_text.append(f"● {state.status} ", style="bold white")
        status_desc = ""

    header_table = Table.grid(expand=True)
    header_table.add_column(ratio=1)
    header_table.add_column(justify="right")
    header_table.add_row(
        status_text + Text(status_desc, style="dim white"),
        Text(state.mode_str, style="bold magenta"),
    )

    # 2. Audio Gauges (VU-Meters)
    mic_bar = render_vu_bar(state.mic_db, width=18, min_db=-50.0, max_db=0.0)
    spk_bar = render_vu_bar(state.speaker_db, width=18, min_db=-50.0, max_db=0.0)

    meter_table = Table(show_header=True, header_style="bold blue", expand=True, box=rich.box.SIMPLE)
    meter_table.add_column("Channel", style="dim", width=12)
    meter_table.add_column("VU Meter", ratio=1)
    meter_table.add_column("Level", justify="right", width=10)

    meter_table.add_row(
        "Microphone",
        f"[bright_green]{mic_bar}[/]",
        f"{state.mic_db:.1f} dB",
    )
    meter_table.add_row(
        "Speakers",
        f"[bright_cyan]{spk_bar}[/]",
        f"{state.speaker_db:.1f} dB",
    )

    # 3. Transcripts / Dialogue
    dialogue_table = Table(box=rich.box.ROUNDED, expand=True, show_header=False)
    dialogue_table.add_column("Speaker", width=14, style="bold")
    dialogue_table.add_column("Dialogue", ratio=1)

    if not state.transcripts:
        dialogue_table.add_row(
            "[dim]Dialogue[/dim]",
            "[italic dim]Speak into your microphone to start the conversation...[/italic dim]",
        )
    else:
        for role, text in state.transcripts:
            if role == "User":
                dialogue_table.add_row("[green]User:[/green]", f"[bright_white]{text}[/bright_white]")
            elif role == "Assistant":
                dialogue_table.add_row("[cyan]Assistant:[/cyan]", f"[bright_cyan]{text}[/bright_cyan]")
            else:
                dialogue_table.add_row(f"[magenta]{role}:[/magenta]", f"[dim]{text}[/dim]")

    footer_text = Text(
        f"Barge-in events: {state.barge_in_count} | Press Ctrl+C to stop",
        style="dim",
        justify="center",
    )

    content = Group(
        header_table,
        meter_table,
        dialogue_table,
        footer_text,
    )

    return Panel(
        content,
        title="[bold cyan]Realtime Voice Assistant[/bold cyan]",
        subtitle="[dim]Bidirectional Audio Streaming (16kHz in / 24kHz out)[/dim]",
        border_style="cyan",
    )


async def run_pipeline(config: AssistantConfig) -> None:
    """Main async pipeline coordinating mic capture, player, and live client."""
    mode_str = f"MOCK ({config.mock_type})" if config.mock else f"LIVE ({config.model})"
    ui_state = AssistantUIState(mode_str=mode_str)

    capture = AudioCapture(
        sample_rate=config.input_sample_rate,
        channels=1,
        block_size=config.block_size,
    )
    player = AudioPlayer(
        sample_rate=config.output_sample_rate,
        channels=1,
        block_size=config.block_size,
    )
    client = create_client(config)

    # Start audio devices
    try:
        player.start()
        await capture.start()
    except MicrophonePermissionError as exc:
        console.print(
            Panel(
                f"[bold red]Microphone Access Denied on macOS[/bold red]\n\n"
                f"{exc}\n\n"
                "[bold yellow]How to fix:[/bold yellow]\n"
                "1. Open [bold]System Settings -> Privacy & Security -> Microphone[/bold].\n"
                "2. Ensure your terminal application (Terminal, iTerm2, VSCode, etc.) is allowed.\n"
                "3. Restart the terminal and try again.",
                title="[red]Audio Device Error[/red]",
                border_style="red",
            )
        )
        player.stop()
        return
    except (MicrophoneError, PlayerError) as exc:
        console.print(f"[bold red]Audio Hardware Error:[/bold red] {exc}")
        player.stop()
        return

    # Connect to live service
    try:
        await client.connect()
    except GeminiLiveProtocolError as exc:
        console.print(
            Panel(
                f"[bold red]Gemini Live Connection Failed[/bold red]\n\n{exc}",
                border_style="red",
            )
        )
        await capture.stop()
        player.stop()
        return

    stop_event = asyncio.Event()

    # Capture upstream task (Mic -> Client)
    async def capture_loop():
        barge_in_active_frames = 0
        while not stop_event.is_set():
            try:
                chunk = await capture.read_chunk()
                db = capture.get_current_db()
                ui_state.mic_db = db

                # Determine if user is speaking based on VAD threshold
                is_user_speaking = db > config.vad_threshold_db

                # Local Barge-in detection:
                # If assistant is playing audio AND user speaks loudly
                if config.barge_in and player.is_playing and is_user_speaking:
                    barge_in_active_frames += 1
                    if barge_in_active_frames >= 2:  # debounce ~60ms
                        player.clear()
                        ui_state.status = "BARGE-IN"
                        ui_state.barge_in_count += 1
                else:
                    barge_in_active_frames = 0

                # Send audio upstream to assistant
                await client.send_audio(chunk)

                # Update status
                if player.is_playing:
                    if ui_state.status != "BARGE-IN":
                        ui_state.status = "SPEAKING"
                elif is_user_speaking:
                    ui_state.status = "STREAMING"
                else:
                    ui_state.status = "LISTENING"

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Capture loop exception: %s", exc)
                break

    # Receive downstream task (Client -> Player)
    async def receive_loop():
        try:
            async for event in client.receive():
                if stop_event.is_set():
                    break

                if isinstance(event, AudioReceived):
                    player.write(event.pcm_data)
                    ui_state.status = "SPEAKING"

                elif isinstance(event, Interrupted):
                    # Server notified barge-in interruption
                    player.clear()
                    ui_state.status = "BARGE-IN"
                    ui_state.barge_in_count += 1

                elif isinstance(event, TranscriptReceived):
                    speaker = "User" if event.is_user else "Assistant"
                    ui_state.add_transcript(speaker, event.text)

                elif isinstance(event, TurnComplete):
                    player.flush()

                elif isinstance(event, VoiceActivity):
                    if event.event_type == "START":
                        if config.barge_in and player.is_playing:
                            player.clear()
                            ui_state.status = "BARGE-IN"
                            ui_state.barge_in_count += 1

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Receive loop exception: %s", exc)

    # UI Refresh task
    async def ui_loop(live: Live):
        while not stop_event.is_set():
            ui_state.speaker_db = player.get_current_db()
            live.update(create_ui_renderable(ui_state))
            await asyncio.sleep(0.04)  # ~25 FPS UI refresh

    # Run tasks with Rich Live display
    with Live(create_ui_renderable(ui_state), refresh_per_second=25, console=console) as live:
        t_capture = asyncio.create_task(capture_loop())
        t_receive = asyncio.create_task(receive_loop())
        t_ui = asyncio.create_task(ui_loop(live))

        # Setup OS signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                # Windows fallback
                pass

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            stop_event.set()
        finally:
            stop_event.set()
            t_capture.cancel()
            t_receive.cancel()
            t_ui.cancel()
            await asyncio.gather(t_capture, t_receive, t_ui, return_exceptions=True)
            try:
                await capture.stop()
            except Exception:
                pass
            try:
                player.stop()
            except Exception:
                pass
            try:
                await client.close()
            except Exception:
                pass

    console.print("\n[bold green]Realtime Voice Assistant cleanly stopped.[/bold green]")


@app.command()
def main(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="Gemini API Key (or set GEMINI_API_KEY environment variable)",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        "-m",
        help="Run in offline simulation mode without an API key (verifies audio devices)",
    ),
    mock_type: str = typer.Option(
        "echo",
        "--mock-type",
        help="Simulation response type: 'echo' (echoes your voice at 24kHz) or 'sine' (audio tones)",
    ),
    model: str = typer.Option(
        "gemini-3.8-live",
        "--model",
        help="Gemini Multimodal Live API model identifier",
    ),
    voice: str = typer.Option(
        "Puck",
        "--voice",
        help="Prebuilt assistant voice name: Puck, Charon, Kore, Fenrir, Aoede",
    ),
    vad_threshold: float = typer.Option(
        -36.0,
        "--vad-threshold",
        help="Microphone volume threshold in dB to trigger voice activity",
    ),
    barge_in: bool = typer.Option(
        True,
        "--barge-in/--no-barge-in",
        help="Enable instant interruption of assistant audio when user starts speaking",
    ),
    list_devices: bool = typer.Option(
        False,
        "--list-devices",
        "-l",
        help="List available audio input and output devices and exit",
    ),
) -> None:
    """Production-ready real-time two-way voice streaming assistant with Gemini Live API."""
    if list_devices:
        console.print("[bold cyan]Available Audio Devices:[/bold cyan]\n")
        devices = sd.query_devices()
        table = Table(box=rich.box.SIMPLE_HEAVY)
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("In Channels", justify="center")
        table.add_column("Out Channels", justify="center")
        table.add_column("Default", justify="center")

        default_in, default_out = sd.default.device
        for idx, dev in enumerate(devices):
            is_default = []
            if idx == default_in:
                is_default.append("INPUT")
            if idx == default_out:
                is_default.append("OUTPUT")
            default_str = ", ".join(is_default) if is_default else "-"

            table.add_row(
                str(idx),
                dev["name"],
                str(dev["max_input_channels"]),
                str(dev["max_output_channels"]),
                f"[green]{default_str}[/]" if default_str != "-" else "[dim]-[/]",
            )
        console.print(table)
        return

    # Validate mock_type
    if mock_type not in ("echo", "sine"):
        console.print("[bold red]Invalid --mock-type. Must be 'echo' or 'sine'.[/bold red]")
        sys.exit(1)

    # Initialize configuration
    config = AssistantConfig.from_env_and_args(
        api_key=api_key,
        mock=mock,
        mock_type=mock_type,
        model=model,
        voice_name=voice,
        vad_threshold_db=vad_threshold,
        barge_in=barge_in,
    )

    # Warn if no API key and not running in mock mode
    if not config.mock and not config.api_key:
        console.print(
            Panel(
                "[bold yellow]No GEMINI_API_KEY detected.[/bold yellow]\n\n"
                "To connect to Google's live server, provide an API key:\n"
                "  $ export GEMINI_API_KEY='your-key'\n"
                "  $ realtime-voice-assistant\n\n"
                "Or run in offline verification mode without an API key:\n"
                "  [bold cyan]$ realtime-voice-assistant --mock[/bold cyan]",
                title="[yellow]Missing API Key[/yellow]",
                border_style="yellow",
            )
        )
        sys.exit(1)

    try:
        asyncio.run(run_pipeline(config))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    app()
