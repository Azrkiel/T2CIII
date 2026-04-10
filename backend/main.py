"""
FastAPI Orchestration Layer for Text-to-CAD Pipeline.

Exposes the hierarchical subagent pipeline as a streaming SSE endpoint.
Each pipeline step emits a JSON progress event so the frontend can render
a live checklist.
"""

import base64
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator

from contextlib import asynccontextmanager

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agents import (
    ScriptError,
    cleanup_tmp_parts,
    is_single_part,
    run_assembly_critic_loop,
    run_critic_loop,
    run_planner,
    run_single_part_export,
)
from compiler import execute_cad_script

logger = logging.getLogger("mirum.api")

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not _GEMINI_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Server cannot start without a valid API key."
    )
genai.configure(api_key=_GEMINI_KEY)


def _safe_error(detail: str | Exception) -> str:
    """Sanitize an error message before returning it to clients."""
    msg = str(detail)
    msg = re.sub(r'File ".*?"', 'File "<script>"', msg)
    msg = re.sub(r"(/app/|/opt/|C:\\\\|C:/)[^\s\"']+", "<internal>", msg)
    return msg

@asynccontextmanager
async def _lifespan(application: FastAPI):
    cleanup_tmp_parts()
    logger.info("Startup: cleared stale tmp_parts")
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Text-to-CAD API", version="0.1.0", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:8501,http://frontend:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def audit_log(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        "%s %s %s %d %.3fs",
        request.client.host if request.client else "-",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=5000)


class RunRequest(BaseModel):
    script: str = Field(..., max_length=50000)


def _event(step: str, status: str, detail: str = "") -> str:
    """Format a server-sent event line."""
    payload = {"step": step, "status": status}
    if detail:
        payload["detail"] = detail
    return f"data: {json.dumps(payload)}\n\n"


async def _pipeline(prompt: str) -> AsyncGenerator[str, None]:
    """Run the full pipeline, yielding SSE progress events."""

    # --- Step 1: Planner ---
    yield _event("planner", "running")
    try:
        manifest = await run_planner(prompt)
    except Exception as e:
        logger.exception("Planner failed")
        yield _event("planner", "error", _safe_error(e))
        return
    part_names = [p.part_id for p in manifest.parts]
    single = is_single_part(manifest)
    mode = "single-part" if single else f"{len(manifest.parts)}-part assembly"
    yield _event("planner", "done", f"Planned {mode}: {', '.join(part_names)}")

    # --- Step 2: Machinist + Critic Loop ---
    output_filename = uuid.uuid4().hex + ".glb"

    if single:
        # Single-part fast path: one machinist call, direct export, no assembler
        part = manifest.parts[0]
        label = f"machinist:{part.part_id}"
        yield _event(label, "running", "Manufacturing single part")
        try:
            code, _ = await run_critic_loop(part)
        except ScriptError as e:
            logger.exception("Machinist failed (single-part)")
            yield _event(label, "error", _safe_error(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": _safe_error(e),
                "script": e.script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        except RuntimeError as e:
            logger.exception("Machinist failed (single-part)")
            yield _event(label, "error", _safe_error(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": _safe_error(e),
                "script": getattr(e, "script", ""),
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        yield _event(label, "done", f"Part '{part.part_id}' validated")

        # Direct export — no assembler needed
        yield _event("export", "running", "Exporting to .glb")
        final_script = await run_single_part_export(code, output_filename)
        yield _event("script", "done", final_script)
        try:
            result = await execute_cad_script(final_script)
        except Exception as e:
            result = {"status": "error", "traceback": str(e)}
        if result["status"] == "error":
            safe_tb = _safe_error(result["traceback"])
            yield _event("export", "error", safe_tb)
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": safe_tb,
                "script": final_script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
    else:
        # Multi-part path: concurrent manufacturing + assembler
        import asyncio

        # Launch all machinist jobs concurrently
        parts = manifest.parts
        labels = {p.part_id: f"machinist:{p.part_id}" for p in parts}

        # Emit "running" for all parts
        for i, part in enumerate(parts, 1):
            yield _event(labels[part.part_id], "running", f"Manufacturing part {i}/{len(parts)}")

        # Run concurrently with semaphore-based rate limiting
        async def _build_part(part):
            code, step_path = await run_critic_loop(part)
            return part.part_id, code, step_path

        tasks = [_build_part(part) for part in parts]
        part_scripts: dict[str, str] = {}
        step_files: dict[str, str] = {}
        for coro in asyncio.as_completed(tasks):
            try:
                part_id, code, step_path = await coro
                part_scripts[part_id] = code
                step_files[part_id] = step_path
                yield _event(labels[part_id], "done", f"Part '{part_id}' validated")
            except ScriptError as e:
                logger.exception("Machinist failed (multi-part)")
                yield _event("machinist", "error", _safe_error(e))
                error_complete = {
                    "step": "complete",
                    "status": "error",
                    "message": _safe_error(e),
                    "script": e.script,
                }
                yield f"data: {json.dumps(error_complete)}\n\n"
                return
            except RuntimeError as e:
                logger.exception("Machinist failed (multi-part)")
                yield _event("machinist", "error", _safe_error(e))
                error_complete = {
                    "step": "complete",
                    "status": "error",
                    "message": _safe_error(e),
                    "script": getattr(e, "script", ""),
                }
                yield f"data: {json.dumps(error_complete)}\n\n"
                return

        # Deterministic assembler (no LLM, no retries)
        yield _event("assembler", "running", "Assembling from pre-compiled .step files")
        try:
            final_script = await run_assembly_critic_loop(manifest, step_files)
            final_script = final_script.replace("output.glb", output_filename)
        except ScriptError as e:
            logger.exception("Assembler failed")
            yield _event("assembler", "error", _safe_error(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": _safe_error(e),
                "script": e.script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        except RuntimeError as e:
            logger.exception("Assembler failed")
            yield _event("assembler", "error", _safe_error(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": _safe_error(e),
                "script": getattr(e, "script", ""),
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return

        yield _event("script", "done", final_script)
        try:
            result = await execute_cad_script(final_script)
        except Exception as e:
            result = {"status": "error", "traceback": str(e)}
        if result["status"] == "error":
            safe_tb = _safe_error(result["traceback"])
            yield _event("assembler", "error", safe_tb)
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": safe_tb,
                "script": final_script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        yield _event("assembler", "done")

    # --- Final: Check file and deliver ---
    if not os.path.exists(output_filename):
        msg = "Script succeeded but .glb file was not created."
        yield _event("export", "error", msg)
        error_complete = {
            "step": "complete",
            "status": "error",
            "message": msg,
            "script": final_script,
        }
        yield f"data: {json.dumps(error_complete)}\n\n"
        return

    if single:
        yield _event("export", "done")

    with open(output_filename, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(output_filename)

    complete_payload = {
        "step": "complete",
        "status": "done",
        "glb": glb_b64,
        "script": final_script,
    }
    yield f"data: {json.dumps(complete_payload)}\n\n"


@app.post("/generate")
@limiter.limit("5/minute")
async def generate(request: GenerateRequest, http_request: Request):
    return StreamingResponse(
        _pipeline(request.prompt),
        media_type="text/event-stream",
    )


@app.post("/run")
@limiter.limit("10/minute")
async def run_script(request: RunRequest, http_request: Request):
    """Execute a CadQuery script directly and return the GLB model."""
    output_filename = uuid.uuid4().hex + ".glb"
    script = request.script.replace("output.glb", output_filename)

    result = await execute_cad_script(script)
    if result["status"] == "error":
        return {"status": "error", "detail": _safe_error(result["traceback"])}

    if not os.path.exists(output_filename):
        return {"status": "error", "detail": "Script ran but no .glb file was created."}

    with open(output_filename, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(output_filename)

    return {"status": "ok", "glb": glb_b64}
