from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import urllib.error
import urllib.request

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

@dataclass(frozen=True)
class DialogueLine:
    speaker: str
    text: str

def _event_parts(event: Any) -> tuple[str, str, dict[str, Any]]:
    if isinstance(event, dict):
        return str(event.get("event_type", "unknown")), str(event.get("description") or event.get("narration_text") or ""), dict(event.get("data") or {})
    return str(getattr(event, "event_type", "unknown")), str(getattr(event, "description", "") or getattr(event, "narration_text", "")), dict(getattr(event, "data", {}) or {})

def _team_name(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value in {"aliado", "ally", "magenta", "equipo_magenta"}: return "la escuadra magenta"
    if value in {"rival", "enemy", "enemigo", "azul", "blue", "equipo_azul"}: return "la escuadra azul"
    return "uno de los equipos"

def template_dialogue(event: Any) -> list[DialogueLine]:
    event_type, description, data = _event_parts(event)
    robot = str(data.get("robot_name") or data.get("display_name") or data.get("robot_id") or "el robot")
    robot_a = str(data.get("robot_a_name") or data.get("robot_a") or "un robot")
    robot_b = str(data.get("robot_b_name") or data.get("robot_b") or "otro robot")
    team = _team_name(data.get("scoring_team") or data.get("team"))
    side_raw = str(data.get("goal_side_field") or data.get("goal_side_image") or "").strip()
    side = "la portería" if not side_raw else (side_raw if "portería" in side_raw.lower() else f"la portería {side_raw}")
    templates = {
        "goal": [DialogueLine("MARTINOLI", f"¡Gol de {team}! La pelota terminó dentro de {side}."), DialogueLine("DOCTOR", "La jugada queda registrada; ahora hay que revisar la trayectoria y la confirmación visual.")],
        "red_card_robot_removed": [DialogueLine("MARTINOLI", f"¡Tarjeta roja! El árbitro retira a {robot}."), DialogueLine("DOCTOR", "Decisión severa, pero el sistema detectó la intervención y la salida del robot.")],
        "robot_grabbed_by_referee": [DialogueLine("MARTINOLI", f"El árbitro interviene y retira a {robot}."), DialogueLine("DOCTOR", "La acción queda marcada para revisión como posible tarjeta roja.")],
        "robot_collision_candidate": [DialogueLine("MARTINOLI", f"¡Choque en la cancha entre {robot_a} y {robot_b}!"), DialogueLine("DOCTOR", "Hay contacto o proximidad extrema; el evento se conserva como posible colisión.")],
        "ball_out_of_field": [DialogueLine("MARTINOLI", "¡La pelota salió de la superficie de juego!"), DialogueLine("DOCTOR", "Se registra balón fuera y queda pendiente el reinicio correspondiente.")],
        "ball_missing_candidate": [DialogueLine("MARTINOLI", "La pelota desaparece momentáneamente de la imagen."), DialogueLine("DOCTOR", "Puede ser una oclusión; el seguimiento mantiene la incertidumbre sin inventar posición.")],
        "ball_recovered": [DialogueLine("MARTINOLI", "¡La pelota vuelve a aparecer!"), DialogueLine("DOCTOR", "El seguimiento recuperó la detección y enlazó nuevamente la trayectoria.")],
        "possession_change": [DialogueLine("MARTINOLI", f"{robot} se queda con la pelota."), DialogueLine("DOCTOR", "Cambio de posesión confirmado por cercanía y continuidad temporal.")],
        "robot_inactive_candidate": [DialogueLine("MARTINOLI", f"Atención con {robot}, dejó de verse o permanece inactivo."), DialogueLine("DOCTOR", "Puede ser oclusión, salida del encuadre o intervención; conviene revisar el contexto.")],
        "robot_reactivated": [DialogueLine("MARTINOLI", f"{robot} vuelve a la acción."), DialogueLine("DOCTOR", "El sistema recuperó su trayectoria después de la interrupción.")],
        "robot_entered_penalty_area": [DialogueLine("MARTINOLI", f"{robot} entra al área penal."), DialogueLine("DOCTOR", "La posición métrica de la cancha confirma la invasión del área.")],
        "referee_intervention_candidate": [DialogueLine("MARTINOLI", "El árbitro mete la mano en la jugada."), DialogueLine("DOCTOR", "Intervención detectada; el evento debe revisarse junto con el robot o la pelota afectados.")],
    }
    if event_type in templates: return templates[event_type]
    clean = description.rstrip(".") or event_type.replace("_", " ")
    return [DialogueLine("MARTINOLI", clean + "."), DialogueLine("DOCTOR", "El evento queda registrado por el sistema de visión.")]

def _load_json_key(path: Path | None) -> str:
    if path is None or not path.exists(): return ""
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return ""
    return str(data.get("GROQ_API_KEY") or data.get("GROQ_API") or "").strip()

def load_groq_api_key(api_config_path: str | Path | None = None) -> str:
    for name in ("GROQ_API_KEY", "GROQ_API"):
        value = os.getenv(name, "").strip()
        if value: return value
    candidates = []
    if api_config_path: candidates.append(Path(api_config_path).expanduser())
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend([project_root / "api.json", project_root / "config" / "api.json", Path.cwd() / "api.json"])
    for candidate in candidates:
        key = _load_json_key(candidate)
        if key: return key
    return ""

def _parse_dialogue(text: str) -> list[DialogueLine]:
    lines = []
    for raw in text.splitlines():
        raw = raw.strip().lstrip("-*• ")
        match = re.match(r"^(MARTINOLI|DOCTOR)\s*:\s*(.+)$", raw, flags=re.IGNORECASE)
        if match: lines.append(DialogueLine(match.group(1).upper(), match.group(2).strip()))
    return lines[:4]

def groq_dialogue(event: Any, *, api_key: str, model: str = DEFAULT_GROQ_MODEL, timeout_seconds: float = 30.0) -> list[DialogueLine]:
    event_type, description, data = _event_parts(event)
    prompt = ("Genera un diálogo breve y deportivo en español para un partido de robots. Usa exactamente los prefijos MARTINOLI: y DOCTOR:. Máximo cuatro líneas. No inventes goles, tarjetas o nombres que no estén en los datos. MARTINOLI es enérgico; DOCTOR es analítico.\n\n" f"Evento: {event_type}\nDescripción: {description}\nDatos: {json.dumps(data, ensure_ascii=False)}")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.65, "max_tokens": 240}).encode("utf-8")
    request = urllib.request.Request(GROQ_URL, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response: payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
    except Exception as exc: raise RuntimeError(f"No fue posible generar el guion con Groq: {exc}") from exc
    parsed = _parse_dialogue(str(text))
    if not parsed: raise RuntimeError("Groq respondió sin líneas MARTINOLI:/DOCTOR: válidas.")
    return parsed

def generate_dialogue(event: Any, *, script_engine: str = "template", api_config_path: str | Path | None = None, groq_model: str = DEFAULT_GROQ_MODEL) -> tuple[list[DialogueLine], str]:
    engine = str(script_engine or "template").strip().lower()
    if engine not in {"template", "groq", "auto"}: raise ValueError("script_engine debe ser template, groq o auto.")
    key = load_groq_api_key(api_config_path)
    if engine == "groq" and not key: raise RuntimeError("Se pidió Groq, pero no hay GROQ_API_KEY/GROQ_API.")
    if engine in {"groq", "auto"} and key:
        try: return groq_dialogue(event, api_key=key, model=groq_model), "groq"
        except RuntimeError as exc:
            if engine == "groq": raise
            print(f"[narración] Groq no disponible; se usarán plantillas: {exc}")
    return template_dialogue(event), "template"
