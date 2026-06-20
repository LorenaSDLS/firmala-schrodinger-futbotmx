"""Compatibility wrapper for the original standalone narration demo.

The production pipeline uses ``src.G_narration.run``. This class exists so old
scripts using ``from narration import Narrator`` continue to work.
"""
from __future__ import annotations
from pathlib import Path
from pydub import AudioSegment
from src.G_narration.dialogue import generate_dialogue
from src.G_narration.tts import synthesize

class Narrator:
    def __init__(self, groq_api_route=None, voice="EDGE", event=None, audio_output_path="salida/narracion.wav", mode="DUETO", toggle_offline=False):
        self.api_config=groq_api_route
        self.voice_mode=str(voice).upper()
        self.event=event or {}
        self.output=Path(audio_output_path)
        self.mode=str(mode).upper()
        self.offline=bool(toggle_offline)
    def narrar_evento_partido(self, salida_final=None):
        out=Path(salida_final or self.output);out.parent.mkdir(parents=True,exist_ok=True)
        event={"event_type":self.event.get("evento_tipo") or self.event.get("event_type") or "evento","description":self.event.get("detalles") or self.event.get("description") or "Evento detectado","data":self.event.get("data") or {}}
        lines,_=generate_dialogue(event,script_engine="template" if self.offline else "auto",api_config_path=self.api_config)
        if self.mode not in {"DUETO","DUO"}:lines=lines[:1]
        engine={"EDGE":"edge","LOQUENDO":"loquendo","WINDOWS":"windows","OFFLINE":"espeak","ESPEAK":"espeak"}.get(self.voice_mode,"edge")
        primary="Jorge" if engine=="loquendo" else "es-MX-JorgeNeural" if engine=="edge" else "es"
        secondary="Carlos" if engine=="loquendo" else "es-US-AlonsoNeural" if engine=="edge" else "es"
        audio=AudioSegment.silent(duration=0,frame_rate=48000).set_channels(1);temps=[]
        for i,line in enumerate(lines):
            temp=out.with_name(f"{out.stem}_line_{i}.wav");temps.append(temp);synthesize(line.text,temp,engine=engine,voice=primary if line.speaker=="MARTINOLI" else secondary,pitch=58 if line.speaker=="MARTINOLI" else 38)
            if len(audio):audio+=AudioSegment.silent(duration=150,frame_rate=48000).set_channels(1)
            audio+=AudioSegment.from_file(temp).set_frame_rate(48000).set_sample_width(2).set_channels(1)
        audio.export(out,format="wav",parameters=["-acodec","pcm_s16le"])
        for temp in temps:
            try:temp.unlink()
            except OSError:pass
        return str(out)
