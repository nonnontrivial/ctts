import json
import typing as t
from dataclasses import asdict, dataclass
from subprocess import Popen

from .user import User

SHORTCUTS = "shortcuts"
SHORTCUTS_CMD = "run"
SHORTCUT_NAME = "send-imessage"


@dataclass
class Message:
    message: str
    recipients: t.List[str]


@dataclass
class SiteSummary:
    name: str
    astro_twilight: str
    predicted_y: float


def build_imessage(user: User, site_summaries: t.List[SiteSummary]) -> Message:
    formatted_site_summaries = "\n".join(
        [
            f"\nsite '{x.name}' predicted to have {x.predicted_y:.2f}mpsas at {x.astro_twilight} UTC."
            for x in site_summaries
        ]
    )
    num_summaries = len(site_summaries)
    message = f"Hi. CTTS predicted {num_summaries} site{'s' if num_summaries > 1 else ''} will have good sky brightness at nearest astronomical twilight:\n{formatted_site_summaries}"
    return Message(message=message, recipients=[user.number])


def send_imessage_to_user(message: Message) -> None:
    """Send an iMessage to a recipient via Shortcuts.

    See https://github.com/kevinschaich/py-imessage-shortcuts
    """
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as f:
        serialized_message = json.dumps(asdict(message))
        f.write(serialized_message.encode())
        temp_file_path = f.name
    Popen([SHORTCUTS, SHORTCUTS_CMD, SHORTCUT_NAME, "--input-path", temp_file_path])
