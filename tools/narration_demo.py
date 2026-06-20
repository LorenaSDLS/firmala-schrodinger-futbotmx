from narration import Narrator

if __name__ == "__main__":
    evento_prueba = {
        "evento_tipo": "PENAL_ATAJADO",
        "detalles": "Minuto 90. El portero desvía el balón y evita el gol.",
    }
    narrador = Narrator(
        groq_api_route="api.json",
        voice="EDGE",
        event=evento_prueba,
        audio_output_path="salida/prueba_edge.wav",
        mode="DUETO",
        toggle_offline=False,
    )
    print(f"EVENTO: {evento_prueba['evento_tipo']}")
    print("Audio:", narrador.narrar_evento_partido())
