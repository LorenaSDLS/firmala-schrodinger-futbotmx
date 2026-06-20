import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

os.environ.setdefault("MPLBACKEND", "Agg")
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.main_supr import run_full_pipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Evita ejecutar varios análisis pesados simultáneamente.
PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# Estados de los análisis.
JOBS = {}
JOB_OUTPUT_DIRECTORIES = {}
JOB_FILES = {}
JOBS_LOCK = Lock()



def update_job(job_id: str, **changes) -> None:
    """Actualiza de forma segura el progreso de un análisis."""
    with JOBS_LOCK:
        if job_id not in JOBS:
            return

        JOBS[job_id].update(changes)
        JOBS[job_id]["updated_at"] = time.time()


def get_public_job(job_id: str):
    """Devuelve únicamente la información que necesita Unity."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)

        if job is None:
            return None

        return {
            "ok": job["status"] != "failed",
            "job_id": job_id,
            "status": job["status"],
            "step": job["step"],
            "total_steps": job["total_steps"],
            "phase": job["phase"],
            "phase_progress": job["phase_progress"],
            "overall_progress": job["overall_progress"],
            "message": job["message"],
            "current_frame": job.get("current_frame", 0),
            "total_frames": job.get("total_frames", 0),
            "error": job.get("error", ""),
            "result_ready": job["status"] == "completed",
        }
def find_pdf_in_output(output_directory: Path) -> Path | None:
    """
    Busca un PDF generado dentro de la carpeta de salida.
    El reporte actual se guarda en output/report/reporte_final.pdf.
    """
    preferred_paths = [
        output_directory / "report" / "reporte_final.pdf",
        output_directory / "reporte_final.pdf",
        output_directory / "match_report.pdf",
        output_directory / "reporte_futbotmx.pdf",
        output_directory / "reporte.pdf",
        output_directory / "report.pdf",
        output_directory / "analysis_report.pdf",
    ]

    for candidate in preferred_paths:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    pdfs = [
        path
        for path in output_directory.rglob("*.pdf")
        if path.exists() and path.stat().st_size > 0
    ]

    if not pdfs:
        return None

    pdfs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return pdfs[0]


def get_job_file(job_id: str, file_key: str) -> Path | None:
    """
    Obtiene una ruta guardada para un trabajo.
    file_key puede ser: preview, pdf, json.
    """
    with JOBS_LOCK:
        files = JOB_FILES.get(job_id, {})
        file_path = files.get(file_key)

    if not file_path:
        return None

    path = Path(file_path)

    if not path.exists():
        return None

    return path

def execute_pipeline(
    job_id: str,
    video_path: Path,
    base_url: str,
) -> None:
    """
    Ejecuta el pipeline en segundo plano.

    Esta función no bloquea la respuesta de /analizar-video.
    """

    def report_progress(**progress) -> None:
        """
        Este callback será llamado por main_supr.py cada vez
        que cambie el avance.
        """
        update_job(
            job_id,
            status="processing",
            **progress,
        )

    try:
        update_job(
            job_id,
            status="processing",
            step=1,
            total_steps=4,
            phase="Preparando análisis",
            phase_progress=0.0,
            overall_progress=2.0,
            message="El video se recibió correctamente.",
        )

        print(f"[API] Ejecutando pipeline del trabajo {job_id}...")

        resultado = run_full_pipeline(
            video_path=video_path,
            yolo_model="v2",

            field_geometry_enabled=True,
            performance_profile="cpu",
            field_segmentation_stride=90,
            field_debug=False,
            camera_stabilization=True,

            team_mode="none",
            save_tracking_debug=False,
            offline_identity_v5=False,

            sam_mode="LoHa",         # SAM desactivado
            sam_confidence=0.18,
            frame_window=20,
            max_frames=None,

            generate_visual_reports=True,
            generate_narration=False,
            generate_pdf=True,
            generate_sample_video=False,


            fast_mode=True,        # Omite el render antiguo de Mesa
            progress_callback=report_progress,
        )

        json_path = Path(resultado["unity_mesa_json_path"])

        with json_path.open("r", encoding="utf-8") as file:
            datos_json = json.load(file)

        preview_video_path = Path(
            resultado["step_02"]["preview_path"]
        )

        output_directory = preview_video_path.parent
        report_pdf_path = None
        report_result = resultado.get("report")

        if isinstance(report_result, dict):
            raw_pdf_path = report_result.get("pdf_path")
            if raw_pdf_path:
                candidate = Path(raw_pdf_path)
                if candidate.exists():
                    report_pdf_path = candidate

        # Respaldo por si report_result no vino por alguna razón.
        if report_pdf_path is None:
            report_pdf_path = find_pdf_in_output(output_directory)

        with JOBS_LOCK:
            JOB_OUTPUT_DIRECTORIES[job_id] = output_directory
            JOB_FILES[job_id] = {
                "preview": str(preview_video_path),
                "json": str(json_path),
                "pdf": str(report_pdf_path) if report_pdf_path else None,
            }

        datos_json["api_meta"] = {
            "job_id": job_id,
            "input_video": str(video_path),

            "preview_video_url": (
                f"{base_url}/videos/preview/{job_id}"
            ),

            "preview_video_download_url": (
                f"{base_url}/videos/preview/{job_id}?download=1"
            ),

            "report_pdf_url": (
                f"{base_url}/reportes/pdf/{job_id}"
                if report_pdf_path is not None
                else None
            ),

            "report_pdf_download_url": (
                f"{base_url}/reportes/pdf/{job_id}?download=1"
                if report_pdf_path is not None
                else None
            ),

            "report_pdf_status_url": (
                f"{base_url}/reportes/pdf/{job_id}/status"
            ),

            "unity_mesa_json_path": str(json_path),
            "output_directory": str(output_directory),
        }

        api_meta = datos_json["api_meta"]

        resultado_final = {
            "ok": True,
            "message": "Video procesado correctamente",
            "data": datos_json,

            # Alias directos para Unity.
            "preview_video_url": api_meta.get("preview_video_url"),
            "preview_video_download_url": api_meta.get("preview_video_download_url"),
            "report_pdf_url": api_meta.get("report_pdf_url"),
            "report_pdf_download_url": api_meta.get("report_pdf_download_url"),
            "report_pdf_status_url": api_meta.get("report_pdf_status_url"),
        }

        update_job(
            job_id,
            status="completed",
            step=4,
            total_steps=4,
            phase="Análisis terminado",
            phase_progress=100.0,
            overall_progress=100.0,
            message="Los resultados están listos.",
            result=resultado_final,
        )

        print(
            f"[API] Pipeline {job_id} terminado correctamente."
        )

    except Exception as error:
        print(f"[API ERROR] Falló el pipeline {job_id}:")
        traceback.print_exc()

        update_job(
            job_id,
            status="failed",
            phase="Error durante el análisis",
            message=str(error),
            error=str(error),
        )


@app.route("/")
def hola():
    return "Hola Unity"


@app.route("/analizar-video", methods=["POST"])
def analizar_video():
    """
    Recibe el video, crea el trabajo y responde inmediatamente
    con un job_id.
    """
    print("Llegó una petición a /analizar-video")

    if "video" not in request.files:
        return jsonify({
            "ok": False,
            "error": "No se recibió archivo 'video'",
        }), 400

    archivo = request.files["video"]

    if archivo.filename == "":
        return jsonify({
            "ok": False,
            "error": "Nombre de archivo no válido",
        }), 400

    filename = secure_filename(archivo.filename)

    if not filename:
        filename = "video.mp4"

    job_id = uuid4().hex
    video_path = UPLOAD_FOLDER / f"{job_id}_{filename}"

    archivo.save(video_path)

    print(f"[API] Video guardado en: {video_path}")
    print(f"[API] Tamaño: {video_path.stat().st_size} bytes")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "step": 1,
            "total_steps": 4,
            "phase": "Video recibido",
            "phase_progress": 0.0,
            "overall_progress": 2.0,
            "current_frame": 0,
            "total_frames": 0,
            "message": "Esperando el inicio del análisis.",
            "error": "",
            "result": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    base_url = request.host_url.rstrip("/")

    # Inicia el pipeline en segundo plano.
    PIPELINE_EXECUTOR.submit(
        execute_pipeline,
        job_id,
        video_path,
        base_url,
    )

    print(f"[API] Trabajo creado: {job_id}")

    # Unity recibe esto sin esperar a que termine el pipeline.
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "progress_url": f"{base_url}/progreso/{job_id}",
        "result_url": f"{base_url}/resultado/{job_id}",
    }), 202


@app.route("/progreso/<job_id>", methods=["GET"])
def obtener_progreso(job_id):
    """
    Unity consulta este endpoint cada 0.75 segundos.
    """
    job = get_public_job(job_id)

    if job is None:
        return jsonify({
            "ok": False,
            "error": "Trabajo no encontrado",
        }), 404

    return jsonify(job), 200


@app.route("/resultado/<job_id>", methods=["GET"])
def obtener_resultado(job_id):
    """
    Unity consulta este endpoint cuando el progreso llega
    al estado completed.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)

        if job is None:
            return jsonify({
                "ok": False,
                "error": "Trabajo no encontrado",
            }), 404

        status = job["status"]
        result = job.get("result")
        error = job.get("error", "")

    if status == "failed":
        return jsonify({
            "ok": False,
            "error": error,
        }), 500

    if status != "completed" or result is None:
        return jsonify({
            "ok": False,
            "status": status,
            "message": "El análisis todavía no ha terminado.",
        }), 202

    return jsonify(result), 200


@app.route(
    "/videos/preview/<job_id>/<filename>",
    methods=["GET"],
)
def descargar_preview(job_id, filename):
    """
    Permite descargar el preview correspondiente a cada trabajo.
    """
    with JOBS_LOCK:
        output_directory = JOB_OUTPUT_DIRECTORIES.get(job_id)

    if output_directory is None:
        return jsonify({
            "ok": False,
            "error": "Preview no disponible",
        }), 404

    return send_from_directory(
        output_directory,
        filename,
        as_attachment=True,
    )

@app.route("/videos/preview/<job_id>", methods=["GET"])
def obtener_preview_video(job_id):
    """
    Devuelve el quick_preview.mp4 del trabajo.
    Unity puede usar esta URL para reproducirlo o descargarlo.
    """
    preview_path = get_job_file(job_id, "preview")

    if preview_path is None:
        return jsonify({
            "ok": False,
            "error": "Preview no disponible",
        }), 404

    download = request.args.get("download", "0") == "1"

    return send_file(
        preview_path,
        mimetype="video/mp4",
        as_attachment=download,
        download_name="quick_preview.mp4",
        conditional=True,
    )

@app.route("/reportes/pdf/<job_id>/status", methods=["GET"])
def estado_reporte_pdf(job_id):
    """
    Permite que Unity pregunte si el PDF ya existe.
    """
    pdf_path = get_job_file(job_id, "pdf")

    if pdf_path is None:
        return jsonify({
            "ok": True,
            "job_id": job_id,
            "ready": False,
            "message": "El reporte PDF todavía no está disponible.",
        }), 200

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "ready": True,
        "filename": pdf_path.name,
        "report_pdf_url": f"{request.host_url.rstrip('/')}/reportes/pdf/{job_id}",
    }), 200


@app.route("/reportes/pdf/<job_id>", methods=["GET"])
def obtener_reporte_pdf(job_id):
    pdf_path = get_job_file(job_id, "pdf")

    if pdf_path is None:
        return jsonify({
            "ok": False,
            "error": "Reporte PDF no disponible",
        }), 404

    download = request.args.get("download", "1") == "1"

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=download,
        download_name=pdf_path.name,
        conditional=True,
    )


if __name__ == "__main__":
    print("[API] Precargando Ultralytics YOLO...", flush=True)

    from ultralytics import YOLO

    print("[API] Ultralytics cargado correctamente.", flush=True)
    print("[API] Iniciando servidor Flask...", flush=True)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )

