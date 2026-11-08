"""The container only has what the Dockerfile copies: every root module the server imports must be listed there."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def root_modules() -> set[str]:
    return {p.stem for p in ROOT.glob("*.py")}


def imported_root_modules(entry: str, seen: set[str] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    if entry in seen:
        return seen
    seen.add(entry)
    tree = ast.parse((ROOT / f"{entry}.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    for name in names & root_modules():
        imported_root_modules(name, seen)
    return seen


def copied_by_dockerfile() -> set[str]:
    files = set()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY ") and ".py" in line:
            files |= {Path(tok).stem for tok in line.split()[1:-1] if tok.endswith(".py")}
    return files


def test_every_root_module_the_server_needs_is_copied_into_the_image():
    needed = imported_root_modules("server")
    missing = needed - copied_by_dockerfile()
    assert not missing, f"the Dockerfile does not copy: {sorted(missing)}"


def test_the_snapshots_and_the_store_are_in_the_image():
    text = (ROOT / "Dockerfile").read_text()
    assert "COPY RAG_data_layers/ RAG_data_layers/" in text and "COPY events_and_chunking/" in text
    assert list((ROOT / "RAG_data_layers" / "session_snapshots").glob("*.npz"))
    assert "session_snapshots" not in (ROOT / ".dockerignore").read_text()


# ── the Render blueprint: Python service plus the TypeScript gateway in front of it ──

def _blueprint():
    import yaml
    services = yaml.safe_load((ROOT / "render.yaml").read_text())["services"]
    return {s["name"]: s for s in services}


def _env(service):
    return {e["key"]: e for e in service.get("envVars", [])}


def test_the_blueprint_has_the_python_service_and_the_gateway():
    services = _blueprint()
    assert services["f1-qualifying-rag"]["runtime"] == "docker"
    assert services["f1-rag-gateway"]["runtime"] == "node" and services["f1-rag-gateway"]["rootDir"] == "ts"


def test_the_gateway_points_at_the_port_the_python_container_serves_on():
    import json
    gateway, python = _env(_blueprint()["f1-rag-gateway"]), _env(_blueprint()["f1-qualifying-rag"])
    port = re.search(r'"--port",\s*"(\d+)"', (ROOT / "Dockerfile").read_text()).group(1)
    assert gateway["UPSTREAM_URL"]["value"] == f"http://f1-qualifying-rag:{port}"
    assert python["PORT"]["value"] == port


def test_both_services_share_one_generated_gateway_secret():
    gateway, python = _env(_blueprint()["f1-rag-gateway"]), _env(_blueprint()["f1-qualifying-rag"])
    assert gateway["GATEWAY_SECRET"].get("generateValue") is True and "value" not in gateway["GATEWAY_SECRET"]
    assert python["GATEWAY_SECRET"]["fromService"] == {"type": "web", "name": "f1-rag-gateway", "envVarKey": "GATEWAY_SECRET"}


def test_no_secret_is_written_into_the_blueprint():
    text = (ROOT / "render.yaml").read_text()
    assert "sk-ant" not in text
    assert _env(_blueprint()["f1-qualifying-rag"])["ANTHROPIC_API_KEY"].get("sync") is False


def test_the_gateway_start_command_runs_a_script_that_exists_and_what_it_needs_is_installed_in_production():
    import json
    gateway = _blueprint()["f1-rag-gateway"]
    package = json.loads((ROOT / "ts" / "gateway" / "package.json").read_text())
    script = re.search(r"npm run (\w+) --workspace gateway", gateway["startCommand"]).group(1)
    assert script in package["scripts"]
    runner = package["scripts"][script].split()[0]
    assert runner in package["dependencies"], f"{runner} runs the gateway but is not a production dependency"
    assert (ROOT / "ts" / "package-lock.json").exists()              # npm ci needs the lockfile
    assert "npm run build" in gateway["buildCommand"] and "build" in json.loads((ROOT / "ts" / "package.json").read_text())["scripts"]


def test_the_gateway_finds_the_frontend_the_build_command_makes():
    main = (ROOT / "ts" / "gateway" / "src" / "main.ts").read_text()
    assert '"../../web/dist"' in main                                # resolved from ts/gateway/src: ts/web/dist, where vite builds
    assert (ROOT / "ts" / "web" / "vite.config.ts").exists()
