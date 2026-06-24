from __future__ import annotations
from pathlib import Path
import base64, json, mimetypes, re, shutil
from jinja2 import Environment, FileSystemLoader, select_autoescape
from src.H_report.charts import generate_charts
from src.H_report.report_data import build_report_data

REPORT_VERSION = "11.5.2"
REQUIRED_TEMPLATE_MARKER = f'futbot-report-version" content="{REPORT_VERSION}"'
REQUIRED_CHARTS = (
    "mapa_movimiento.png",
    "zona_control_magenta.png",
    "zona_control_azul.png",
    "grafica_posesion.png",
)


def _inline(html: str, base: Path) -> str:
    pattern = re.compile(r'src=["\'](assets/[^"\']+)["\']')

    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        path = base / relative
        if not path.exists():
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{encoded}"'

    return pattern.sub(replace, html)


def _pdf(html_path: Path, pdf_path: Path) -> str:
    """Render fixed-size report pages instead of one infinitely tall PDF page."""
    playwright_error = None
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            # Prefer the Chromium version installed by Playwright. The user's
            # original PDF was produced by local Skia/PDF m145, while the smooth
            # demo was produced by Playwright Chromium m144. Falling back to the
            # system browser keeps the report usable when the bundled browser is
            # not installed, but avoids local-browser PDF regressions by default.
            browser_source = "bundled"
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as bundled_error:
                executable = (
                    shutil.which("chromium")
                    or shutil.which("chromium-browser")
                    or shutil.which("google-chrome")
                )
                if not executable:
                    raise bundled_error
                browser_source = "system"
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=executable,
                )
            page = browser.new_page(viewport={"width": 1048, "height": 1595})
            page.set_content(html_path.read_text(encoding="utf-8"), wait_until="load")
            page.emulate_media(media="print")

            # Espera fuentes e imágenes; evita PDFs incompletos o descentrados
            # cuando Chromium imprime antes de terminar el layout.
            page.evaluate("document.fonts && document.fonts.ready")
            page.wait_for_function(
                "Array.from(document.images).every((image) => image.complete)"
            )
            page.wait_for_timeout(120)
            page.pdf(
                path=str(pdf_path),
                print_background=True,
                prefer_css_page_size=True,
                scale=1,
                display_header_footer=False,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            browser.close()
            return f"playwright-{browser_source}"
    except Exception as exc:  # pragma: no cover - fallback depends on host packages
        playwright_error = exc

    try:
        from weasyprint import HTML

        HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))
        return "weasyprint"
    except Exception as exc:  # pragma: no cover - fallback depends on host packages
        raise RuntimeError(
            f"No se pudo generar PDF. Playwright: {playwright_error}. WeasyPrint: {exc}"
        ) from exc


def _copy_report_asset(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        shutil.copy2(source, target)
        return
    try:
        from PIL import Image

        image = Image.open(source)
        name = source.name.lower()
        if source.parent.name == "transcripcion":
            maximum = (96, 96)
        elif name == "logo.png":
            maximum = (620, 220)
        elif name == "logo_solo.png":
            maximum = (130, 130)
        elif name in {"aliado.png", "enemigo.png"}:
            maximum = (420, 420)
        elif name in {"aliadosf.png", "enemigosf.png"}:
            maximum = (220, 220)
        elif name == "aliado enemigo.png":
            maximum = (360, 240)
        elif name == "cancha.png":
            maximum = (900, 400)
        else:
            maximum = (700, 700)
        image.thumbnail(maximum, Image.Resampling.LANCZOS)
        save_options = {"optimize": True}
        if source.suffix.lower() == ".png":
            save_options["compress_level"] = 7
        image.save(target, **save_options)
    except Exception:
        shutil.copy2(source, target)


def _load_template_source(template_path: Path) -> str:
    source = template_path.read_text(encoding="utf-8")
    if REQUIRED_TEMPLATE_MARKER not in source:
        raise RuntimeError(
            "La plantilla del reporte no corresponde a FutBot V11.5.2. "
            "Se detectó una plantilla antigua que usa campos como Estrategia, Pases, "
            "Tiros o Eficiencia. Reemplaza src/H_report/templates/infographic.html "
            "con la incluida en el hotfix V11.5.2 y vuelve a ejecutar --pdf."
        )
    return source


def _validate_charts(assets: Path) -> None:
    missing = []
    for filename in REQUIRED_CHARTS:
        path = assets / filename
        if not path.exists() or path.stat().st_size < 1000:
            missing.append(filename)
    if missing:
        raise RuntimeError(
            "No se generaron correctamente las gráficas requeridas: " + ", ".join(missing)
        )


def _validate_rendered_html(html: str) -> None:
    if "{{" in html or "{%" in html:
        raise RuntimeError("El HTML final todavía contiene variables Jinja sin resolver.")
    if 'data-event-page="2"' not in html:
        raise RuntimeError("La plantilla no creó la página 2 dedicada a eventos.")
    if "Registro cronológico completo" not in html:
        raise RuntimeError("La plantilla multipágina de eventos no está activa.")
    unresolved = sorted(set(re.findall(r'src=["\'](assets/[^"\']+)["\']', html)))
    if unresolved:
        raise RuntimeError(
            "El reporte contiene imágenes que no pudieron cargarse: " + ", ".join(unresolved)
        )
    legacy_tokens = ("Estrategia: Ofensiva", "Estrategia: Defensiva", ">Pases<", "Eficiencia</div>")
    if any(token in html for token in legacy_tokens):
        raise RuntimeError("Se intentó renderizar una plantilla antigua del reporte.")


def run_report(
    output_directory,
    events_path,
    summary_path,
    tracks_path,
    max_featured_events=8,
    metadata_path=None,
):
    output = Path(output_directory)
    report_directory = output / "report"
    assets = report_directory / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent

    # Never reuse stale charts from a previous report run.
    for filename in REQUIRED_CHARTS:
        (assets / filename).unlink(missing_ok=True)

    for item in (package / "assets").rglob("*"):
        if not item.is_file():
            continue
        target = assets / item.relative_to(package / "assets")
        _copy_report_asset(item, target)

    metadata = Path(metadata_path) if metadata_path else output / "video_metadata.json"
    data = build_report_data(
        events_path,
        summary_path,
        tracks_path,
        max_featured_events,
        metadata_path=metadata,
        output_directory=output,
    )
    chart_paths = generate_charts(
        tracks_path,
        assets,
        team_assignments=data.get("team_assignments"),
    )
    _validate_charts(assets)

    template_path = package / "templates" / "infographic.html"
    template_source = _load_template_source(template_path)
    environment = Environment(
        loader=FileSystemLoader(str(package / "templates")),
        autoescape=select_autoescape(["html"]),
    )
    html = environment.from_string(template_source).render(
        **data,
        qr_available=(assets / "qr.png").exists(),
    )
    html = _inline(html, report_directory)
    _validate_rendered_html(html)

    html_path = report_directory / "reporte_final.html"
    pdf_path = report_directory / "reporte_final.pdf"
    data_path = report_directory / "report_data.json"
    html_path.write_text(html, encoding="utf-8")
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    renderer = _pdf(html_path, pdf_path)

    return {
        "html_path": str(html_path),
        "pdf_path": str(pdf_path),
        "data_path": str(data_path),
        "featured_event_count": max(0, len(data["eventos"]) - 1),
        "event_count": data.get("event_count", len(data["eventos"])),
        "page_count": data.get("total_page_count", 1),
        "renderer": renderer,
        "chart_paths": chart_paths,
        "report_version": REPORT_VERSION,
    }
