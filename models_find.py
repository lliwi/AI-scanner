#!/usr/bin/env python3
"""Busca servidores de IA expuestos (sin autenticación) usando la API de
Shodan y lista, por cada IP, los modelos disponibles.

Soporta varios proveedores (ver PROVIDERS): Ollama y servidores compatibles
con la API de OpenAI (llama.cpp, vLLM, LocalAI, LM Studio, text-gen-webui).

Uso:
    python models_find.py [--limit N] [--no-probe] [--timeout S]
    python models_find.py --update                 # refresca la caché desde Shodan
    python models_find.py --providers ollama,vllm  # solo ciertos proveedores
    python models_find.py --chat                   # chatea usando la caché
    python models_find.py --test                   # prueba y mide velocidad

Los resultados se guardan en hosts.json. En modo --chat (o cualquier
ejecución normal) se reutiliza esa caché si existe; con --update se vuelve a
consultar Shodan y se sobrescribe el fichero. El flujo de chat es
modelo-primero: eliges modelo y luego el servidor que lo ofrece.

La API key se lee de la variable SHODAN_API_KEY (fichero .env o entorno).
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

import shodan

# Servidores de IA que suelen exponerse sin autenticación. Cada proveedor
# define su consulta de Shodan, el puerto por defecto y el "protocolo" con el
# que hablamos con él para listar modelos / chatear / probar:
#   - "ollama": API nativa de Ollama (/api/tags, /api/chat, /api/generate).
#   - "openai": API compatible con OpenAI (/v1/models, /v1/chat/completions).
# Las consultas de Shodan de algunos (vLLM, LM Studio, text-gen-webui) son
# aproximadas; la confirmación real se hace al sondear el host. Edítalas si
# quieres afinar la búsqueda.
PROVIDERS = {
    "ollama": {
        "query": 'product:Ollama',
        "protocol": "ollama",
        "port": 11434,
    },
    "llamacpp": {
        "query": '"Server: llama.cpp"',
        "protocol": "openai",
        "port": 8080,
    },
    "vllm": {
        "query": 'html:"vllm" "HTTP/1.1 200"',
        "protocol": "openai",
        "port": 8000,
    },
    "localai": {
        "query": '"Server: LocalAI"',
        "protocol": "openai",
        "port": 8080,
    },
    "lmstudio": {
        "query": 'port:1234 "Content-Type: application/json" "access-control-allow-origin"',
        "protocol": "openai",
        "port": 1234,
    },
    "tgwebui": {
        "query": 'port:5000 "openai" "HTTP/1.1 200"',
        "protocol": "openai",
        "port": 5000,
    },
}

# Query heredada (solo Ollama), por compatibilidad con la caché antigua.
SHODAN_QUERY = PROVIDERS["ollama"]["query"]

# Fichero donde se cachean los resultados de la búsqueda.
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hosts.json")


def save_cache(hosts: list, path: str = CACHE_FILE):
    """Guarda los hosts en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"query": SHODAN_QUERY, "hosts": hosts}, f,
                  ensure_ascii=False, indent=2)


def load_cache(path: str = CACHE_FILE):
    """Carga los hosts desde el JSON, o None si no existe / está corrupto."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("hosts", [])
    except (json.JSONDecodeError, OSError):
        return None


def env_test_threads(default: int = 10) -> int:
    """Número de hilos para las pruebas de modelos, leído de TEST_THREADS
    (fichero .env o entorno). Si no está o es inválido, usa `default`."""
    load_dotenv()
    try:
        return max(1, int(os.getenv("TEST_THREADS", default)))
    except (TypeError, ValueError):
        return default


def get_test(host: dict, model: str):
    """Devuelve el resultado de --test guardado para (host, modelo), o None."""
    return (host.get("tests") or {}).get(model)


def server_speed_key(host: dict, model: str):
    """Clave de orden para servidores: más rápido primero, luego los OK sin
    métrica (por latencia), y al final los que fallan o no se han probado."""
    t = get_test(host, model)
    if not t or not t.get("ok"):
        return (2, 0.0)
    tps = t.get("tokens_per_sec")
    if tps:
        return (0, -tps)               # rápidos primero
    return (1, t.get("latency") or 9999.0)


def get_models_from_host(ip: str, port: int, timeout: float, protocol: str = "ollama"):
    """Consulta el endpoint de listado de modelos del host según su protocolo.
    Devuelve una lista de nombres, [] si responde pero sin modelos, o None si
    no responde / no es un servidor válido."""
    if protocol == "openai":
        url = f"http://{ip}:{port}/v1/models"
        key = "id"
        container = "data"
    else:  # ollama
        url = f"http://{ip}:{port}/api/tags"
        key = "name"
        container = "models"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m.get(key, "?") for m in data.get(container, [])]
    except Exception:
        return None


def models_from_shodan_banner(match: dict, protocol: str = "ollama"):
    """Intenta extraer los modelos que Shodan ya haya capturado en el banner.
    Ollama usa "name" (en "models"); la API OpenAI usa "id" (en "data")."""
    import re
    data = match.get("data", "") or ""
    field = "id" if protocol == "openai" else "name"
    models = []
    for name in re.findall(rf'"{field}"\s*:\s*"([^"]+)"', data):
        if name not in models:
            models.append(name)
    return models


def _ask(prompt: str):
    """input() que trata Ctrl-C / EOF como cancelación limpia."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def select_from_list(items, render, prompt):
    """Muestra una lista numerada y devuelve el elemento elegido (o None)."""
    for i, it in enumerate(items, 1):
        print(f"  [{i}] {render(it)}")
    while True:
        raw = _ask(f"\n{prompt} (1-{len(items)}, q para salir): ")
        if raw is None or raw.strip().lower() in ("q", "quit", "salir"):
            return None
        if raw.strip().isdigit():
            idx = int(raw.strip())
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("    Selección no válida.")


def chat_stream_iter(host_url: str, model: str, messages: list, timeout: float,
                     protocol: str = "ollama"):
    """Generador que envía la conversación en streaming y va cediendo (yield)
    cada fragmento de texto del asistente. Si hay un error, cede una tupla
    ("__error__", mensaje). Soporta el protocolo nativo de Ollama y el
    compatible con OpenAI. Reutilizable por CLI y TUI."""
    if protocol == "openai":
        url = f"{host_url}/v1/chat/completions"
    else:
        url = f"{host_url}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        yield ("__error__", f"error de conexión: {e}")
        return

    for line in resp.iter_lines():
        if not line:
            continue

        if protocol == "openai":
            # Server-Sent Events: líneas "data: {json}" y "data: [DONE]".
            text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
            if not text.startswith("data:"):
                continue
            text = text[len("data:"):].strip()
            if text == "[DONE]":
                break
            try:
                chunk = json.loads(text)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content", "")
            if piece:
                yield piece
        else:
            # Ollama: una línea JSON por fragmento.
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:
                yield ("__error__", f"error del servidor: {chunk['error']}")
                return
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break


def chat_stream(host_url: str, model: str, messages: list, timeout: float,
                protocol: str = "ollama"):
    """Versión CLI: imprime la respuesta en streaming. Devuelve el texto
    completo del asistente o None si falla."""
    full = []
    try:
        for piece in chat_stream_iter(host_url, model, messages, timeout, protocol):
            if isinstance(piece, tuple):  # ("__error__", msg)
                print(f"    [{piece[1]}]")
                return None
            full.append(piece)
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        print("\n    [respuesta interrumpida]")
    print()
    return "".join(full)


# Categorías estables de error para poder agregar estadísticas. El texto libre
# de `error`/`error_detail` varía entre servidores; `error_category` sí está
# acotado a este conjunto, así que es lo que hay que usar para contar/agrupar.
ERROR_CATEGORIES = {
    "timeout":          "La petición superó el timeout",
    "connection":       "No se pudo conectar (rechazada/DNS/red inalcanzable)",
    "auth":             "Requiere API key o acceso restringido (401/403)",
    "not_found":        "Modelo o endpoint inexistente (404)",
    "rate_limit":       "Límite de peticiones o cuota agotada (429)",
    "model_error":      "El modelo existe pero no se pudo cargar/ejecutar",
    "bad_request":      "Petición rechazada por inválida (400)",
    "server_error":     "Error interno del servidor (5xx)",
    "invalid_response": "Respuesta no válida / no-JSON",
    "other":            "Otro error no clasificado",
}


def _classify_http_status(status: int) -> str:
    """Mapea un código HTTP de error a una categoría estable."""
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limit"
    if status == 400:
        return "bad_request"
    if 500 <= status < 600:
        return "server_error"
    return "other"


def _classify_message(msg: str):
    """Heurística sobre el texto de error del servidor para afinar la causa
    (p. ej. distinguir 'falta API key' de 'modelo no cargado'). Devuelve una
    categoría o None si no reconoce el mensaje."""
    m = (msg or "").lower()
    if not m:
        return None
    if any(k in m for k in ("api key", "api-key", "unauthorized", "authenticat",
                            "forbidden", "permission", "not allowed", "access denied")):
        return "auth"
    if any(k in m for k in ("rate limit", "too many requests", "quota", "overloaded")):
        return "rate_limit"
    if any(k in m for k in ("not found", "no such model", "does not exist",
                            "unknown model", "model not found", "try pulling")):
        return "not_found"
    if any(k in m for k in ("out of memory", "not loaded", "failed to load",
                            "loading model", "runner", "no available", "cuda",
                            "insufficient")):
        return "model_error"
    return None


def _extract_server_error(data):
    """Extrae el mensaje de error embebido en el cuerpo JSON de la respuesta,
    tanto en formato Ollama ({"error": "..."}) como OpenAI
    ({"error": {"message": "..."}}). Devuelve el texto o None."""
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if not err:
        return None
    if isinstance(err, dict):
        return err.get("message") or err.get("code") or str(err)
    return str(err)


def _fail(category: str, error, latency: float, http_status=None, detail=None):
    """Construye un dict de resultado fallido homogéneo."""
    res = {
        "ok": False,
        "error": str(error)[:100],       # resumen legible (retrocompatible)
        "error_category": category,       # valor acotado para estadísticas
        "latency": latency,
    }
    if http_status is not None:
        res["http_status"] = http_status
    if detail:
        res["error_detail"] = str(detail)[:300]  # mensaje crudo del servidor
    return res


def test_model(ip: str, port: int, model: str, timeout: float, num_predict: int,
               protocol: str = "ollama"):
    """Envía una generación mínima a un modelo y mide su rendimiento.

    Devuelve un dict con:
      ok             -> True/False
      latency        -> tiempo total de la petición (s)
      tokens_per_sec -> velocidad de generación (métricas del servidor si las
                        hay; si no, tokens generados / latencia)
      eval_count     -> nº de tokens generados
    Y si ok=False:
      error          -> resumen legible del fallo
      error_category -> categoría estable (ver ERROR_CATEGORIES) para stats
      http_status    -> código HTTP, si el fallo vino con respuesta
      error_detail   -> mensaje crudo del servidor, si lo hubo
    """
    prompt = "Responde solo con la palabra: OK"
    if protocol == "openai":
        url = f"http://{ip}:{port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": num_predict,
        }
    else:
        url = f"http://{ip}:{port}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict},
        }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        return _fail("timeout", "timeout", time.perf_counter() - t0)
    except requests.exceptions.ConnectionError as e:
        return _fail("connection", f"conexión: {type(e).__name__}",
                     time.perf_counter() - t0)
    except Exception as e:
        return _fail("other", type(e).__name__, time.perf_counter() - t0)

    latency = time.perf_counter() - t0

    # El cuerpo puede traer el detalle del error; lo intentamos parsear siempre.
    try:
        data = resp.json()
    except ValueError:
        data = None
    server_msg = _extract_server_error(data)

    # Fallo por código HTTP (>=400). Afinamos con el mensaje si lo reconocemos.
    if not resp.ok:
        category = _classify_message(server_msg) or _classify_http_status(resp.status_code)
        summary = server_msg or f"HTTP {resp.status_code} {resp.reason}"
        return _fail(category, summary, latency,
                     http_status=resp.status_code, detail=server_msg)

    # 200 pero con error embebido (Ollama suele devolver 200 + {"error": ...}).
    if server_msg:
        category = _classify_message(server_msg) or "model_error"
        return _fail(category, server_msg, latency, detail=server_msg)

    if data is None:
        return _fail("invalid_response", "respuesta no-JSON", latency)

    if protocol == "openai":
        # La API OpenAI no da tiempos de servidor; estimamos tok/s con la latencia.
        usage = data.get("usage") or {}
        eval_count = usage.get("completion_tokens") or 0
        tps = (eval_count / latency) if (eval_count and latency) else None
    else:
        eval_count = data.get("eval_count") or 0
        eval_dur = data.get("eval_duration") or 0  # nanosegundos
        tps = (eval_count / (eval_dur / 1e9)) if eval_dur else None
    return {
        "ok": True,
        "latency": latency,
        "tokens_per_sec": tps,
        "eval_count": eval_count,
    }


def run_tests(hosts, timeout: float, num_predict: int, workers: int):
    """Prueba cada (servidor, modelo) en paralelo, guarda los resultados en
    los hosts y muestra un informe. Devuelve la lista de resultados planos."""
    usables = [h for h in hosts if h.get("live_models")]
    pairs = [(h, m) for h in usables for m in h["live_models"]]
    if not pairs:
        print("\n[!] No hay servidores con modelos accesibles para probar.")
        return []

    total_models = len(pairs)
    print("\n" + "=" * 70)
    print(f"MODO TEST — probando {total_models} modelos en {len(usables)} servidores")
    print(f"(num_predict={num_predict}, timeout={timeout}s, {workers} en paralelo)")
    print("=" * 70)

    results = []
    done = 0

    def work(pair):
        h, m = pair
        res = test_model(h["ip"], h["port"], m, timeout, num_predict,
                         h.get("protocol", "ollama"))
        return h, m, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, p) for p in pairs]
        for fut in concurrent.futures.as_completed(futures):
            h, m, res = fut.result()
            done += 1
            # Guardar el resultado en el host (para cachear).
            h.setdefault("tests", {})[m] = res
            results.append({
                "ip": h["ip"], "port": h["port"], "country": h.get("country", ""),
                "model": m, **res,
            })
            if res["ok"]:
                tps = res["tokens_per_sec"]
                tps_s = f"{tps:5.1f} tok/s" if tps else "  ?  tok/s"
                status = f"OK  {tps_s}  {res['latency']:5.1f}s"
            else:
                status = f"FALLO ({res['error']})"
            print(f"  [{done}/{total_models}] {h['ip']}:{h['port']}  {m:<40} {status}")

    # ---- Informe final ----
    ok = [r for r in results if r["ok"]]
    ko = [r for r in results if not r["ok"]]

    print("\n" + "=" * 70)
    print("INFORME")
    print("=" * 70)
    print(f"Funcionan: {len(ok)}   |   Fallan: {len(ko)}   |   Total: {len(results)}")

    if ok:
        # Ranking por velocidad (tokens/s desc; los sin métrica al final).
        ranked = sorted(
            ok,
            key=lambda r: (r["tokens_per_sec"] is None,
                           -(r["tokens_per_sec"] or 0)),
        )
        print("\nTop por velocidad (tokens/segundo):")
        print(f"  {'#':>2}  {'tok/s':>7}  {'latencia':>8}  servidor / modelo")
        for i, r in enumerate(ranked[:15], 1):
            tps = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] else "?"
            print(f"  {i:>2}  {tps:>7}  {r['latency']:>7.1f}s  "
                  f"{r['ip']}:{r['port']}  {r['model']}")

    if ko:
        # Desglose por categoría de error (para estadísticas).
        by_cat = {}
        for r in ko:
            by_cat.setdefault(r.get("error_category", "other"), []).append(r)
        print(f"\nFallos por categoría ({len(ko)} en total):")
        for cat, rows in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            desc = ERROR_CATEGORIES.get(cat, cat)
            print(f"  {len(rows):>4}  {cat:<16} {desc}")

        print(f"\nModelos que fallan ({len(ko)}):")
        for r in ko:
            cat = r.get("error_category", "?")
            status = f" HTTP {r['http_status']}" if r.get("http_status") else ""
            print(f"  - {r['ip']}:{r['port']}  {r['model']}  "
                  f"[{cat}{status}] {r['error']}")

    return results


def start_chat(hosts, default_timeout: float):
    """Flujo interactivo modelo-primero: elegir modelo -> elegir servidor
    (entre los que lo ofrecen) -> chatear."""
    usables = [h for h in hosts if h.get("live_models")]
    if not usables:
        print("\n[!] No hay servidores con modelos accesibles en vivo para chatear.")
        return

    # Agregar: modelo -> lista de hosts que lo ofrecen.
    model_hosts = {}
    for h in usables:
        for name in h["live_models"]:
            model_hosts.setdefault(name, []).append(h)

    # Ordenar modelos por número de servidores que los sirven (desc) y nombre.
    models = sorted(model_hosts.keys(),
                    key=lambda m: (-len(model_hosts[m]), m.lower()))

    def model_label(m):
        n = len(model_hosts[m])
        text = f"{m}  ({n} servidor{'es' if n != 1 else ''})"
        # Verde si algún servidor que lo ofrece ha pasado la prueba (ok:true).
        if any((get_test(h, m) or {}).get("ok") for h in model_hosts[m]):
            return f"\033[32m{text}\033[0m"
        return text

    print("\n" + "=" * 70)
    print(f"MODO CHAT — {len(models)} modelos disponibles en {len(usables)} servidores")
    print("=" * 70)
    model = select_from_list(models, model_label, "¿Qué modelo quieres usar?")
    if model is None:
        print("Cancelado.")
        return

    # Elegir servidor entre los que ofrecen ese modelo, ordenados por
    # velocidad (más rápido primero según los datos de --test).
    candidates = sorted(model_hosts[model], key=lambda h: server_speed_key(h, model))

    def server_label(h):
        base = f"{h['ip']}:{h['port']}"
        loc = f" · {h['country']}" if h.get("country") else ""
        t = get_test(h, model)
        if t is None:
            return f"{base}  (sin probar){loc}"
        if not t.get("ok"):
            # Rojo para los que fallan.
            return f"\033[31m{base}  FALLO ({t.get('error', '?')})\033[0m{loc}"
        tps = t.get("tokens_per_sec")
        speed = f"{tps:.1f} tok/s" if tps else "? tok/s"
        # Verde + latencia para los que funcionan.
        return f"\033[32m{base}\033[0m  {speed} · {t.get('latency', 0):.1f}s{loc}"

    if len(candidates) == 1:
        host = candidates[0]
        print(f"\nÚnico servidor con {model}: {server_label(host)}")
    else:
        print(f"\nServidores que ofrecen {model} (más rápido primero):")
        host = select_from_list(candidates, server_label, "¿Qué servidor usar?")
        if host is None:
            print("Cancelado.")
            return

    host_url = f"http://{host['ip']}:{host['port']}"
    protocol = host.get("protocol", "ollama")
    # Timeout más generoso para la generación que para el sondeo inicial.
    chat_timeout = max(default_timeout, 120.0)

    print("\n" + "=" * 70)
    print(f"Chat con {model} @ {host['ip']}:{host['port']} "
          f"[{host.get('provider', 'ollama')}]")
    print("Escribe tu mensaje. Comandos: /reset (limpiar contexto), /salir")
    print("=" * 70)

    messages = []
    while True:
        user_msg = _ask("\n> ")
        if user_msg is None:
            break
        user_msg = user_msg.strip()
        if not user_msg:
            continue
        if user_msg.lower() in ("/salir", "/exit", "/quit"):
            break
        if user_msg.lower() == "/reset":
            messages = []
            print("    [contexto reiniciado]")
            continue

        messages.append({"role": "user", "content": user_msg})
        print()
        reply = chat_stream(host_url, model, messages, chat_timeout, protocol)
        if reply is None:
            # Deshacer el último mensaje para poder reintentar.
            messages.pop()
        else:
            messages.append({"role": "assistant", "content": reply})

    print("\nSesión de chat finalizada.")


def fetch_from_shodan(limit: int, no_probe: bool, timeout: float, providers=None):
    """Consulta Shodan para cada proveedor seleccionado, sondea los hosts en
    vivo y devuelve una lista unificada de hosts (con su protocolo/proveedor).

    `providers` es una lista de claves de PROVIDERS; None = todos."""
    load_dotenv()
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        sys.exit("ERROR: no se encontró SHODAN_API_KEY en el entorno ni en .env")

    api = shodan.Shodan(api_key)
    selected = providers or list(PROVIDERS.keys())

    hosts = []
    seen = set()  # dedup por ip:port
    for name in selected:
        prov = PROVIDERS.get(name)
        if not prov:
            print(f"[!] Proveedor desconocido: {name} (ignorado)")
            continue
        query, protocol = prov["query"], prov["protocol"]
        print(f"[*] [{name}] Buscando en Shodan: {query!r}")
        try:
            results = api.search(query, limit=limit)
        except shodan.APIError as e:
            print(f"    [!] Error de Shodan para {name}: {e} (se omite)")
            continue

        matches = results.get("matches", [])
        print(f"    total {results.get('total', 0)}, procesando {len(matches)}")
        for m in matches:
            ip, port = m.get("ip_str"), m.get("port", prov["port"])
            key = f"{ip}:{port}"
            if key in seen:
                continue
            seen.add(key)
            hosts.append({
                "ip": ip,
                "port": port,
                "provider": name,
                "protocol": protocol,
                "org": m.get("org", ""),
                "country": (m.get("location", {}) or {}).get("country_name", ""),
                "shodan_models": models_from_shodan_banner(m, protocol),
            })

    print(f"\n[*] {len(hosts)} servidores únicos encontrados.\n")

    # Consultar modelos en vivo (en paralelo) salvo que se pida --no-probe
    if not no_probe:
        def probe(h):
            h["live_models"] = get_models_from_host(
                h["ip"], h["port"], timeout, h.get("protocol", "ollama"))
            return h

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            hosts = list(ex.map(probe, hosts))
    else:
        for h in hosts:
            h["live_models"] = None

    return hosts


def print_listing(hosts: list):
    """Imprime el listado de servidores y sus modelos."""
    print("=" * 70)
    activos = 0
    for h in hosts:
        models = h["live_models"] if h["live_models"] is not None else h["shodan_models"]
        estado = "activo" if h["live_models"] else ("shodan" if h["shodan_models"] else "sin datos")
        if models:
            activos += 1
        loc = ", ".join(x for x in (h["country"], h["org"]) if x)
        prov = h.get("provider", "ollama")
        print(f"\nIP: {h['ip']}:{h['port']}  [{prov}] [{estado}]")
        if loc:
            print(f"    {loc}")
        if models:
            for name in models:
                print(f"    - {name}")
        else:
            print("    (sin modelos accesibles)")

    print("\n" + "=" * 70)
    print(f"[*] {len(hosts)} servidores procesados, {activos} con modelos listados.")


def main():
    parser = argparse.ArgumentParser(
        description="Busca servidores de IA (Ollama, vLLM, llama.cpp…) con Shodan")
    parser.add_argument("--limit", type=int, default=100,
                        help="Número máximo de resultados a procesar (def: 100)")
    parser.add_argument("--no-probe", action="store_true",
                        help="No consultar los hosts en vivo; usar solo datos de Shodan")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout en segundos para consultar cada host (def: 5)")
    parser.add_argument("--chat", action="store_true",
                        help="Elegir modelo y servidor e iniciar un chat")
    parser.add_argument("--update", action="store_true",
                        help="Forzar nueva consulta a Shodan y sobrescribir la caché")
    parser.add_argument("--test", action="store_true",
                        help="Probar cada servidor/modelo y medir su velocidad")
    parser.add_argument("--test-timeout", type=float, default=60.0,
                        help="Timeout por prueba de modelo en --test (def: 60)")
    parser.add_argument("--num-predict", type=int, default=16,
                        help="Tokens a generar en cada prueba de --test (def: 16)")
    parser.add_argument("--test-workers", type=int, default=None,
                        help="Pruebas en paralelo en --test "
                             "(def: TEST_THREADS del .env, o 10)")
    parser.add_argument("--providers", default="all",
                        help="Proveedores a buscar, separados por comas "
                             f"({', '.join(PROVIDERS)}) o 'all' (def: all)")
    args = parser.parse_args()

    if args.providers.strip().lower() == "all":
        providers = list(PROVIDERS.keys())
    else:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    if args.chat and args.no_probe:
        print("[!] --chat requiere sondear los hosts; ignorando --no-probe.")
        args.no_probe = False

    cached = load_cache()

    # Decidir de dónde salen los datos:
    #  - --update, o no hay caché  -> consultar Shodan y guardar.
    #  - en otro caso              -> usar la caché existente.
    if args.update or cached is None:
        if cached is None and not args.update:
            print("[*] No existe caché; consultando Shodan por primera vez.")
        hosts = fetch_from_shodan(args.limit, args.no_probe, args.timeout, providers)
        save_cache(hosts)
        print(f"[*] Resultados guardados en {CACHE_FILE}")
    else:
        hosts = cached
        print(f"[*] Usando caché de {CACHE_FILE} ({len(hosts)} servidores). "
              f"Usa --update para refrescar.")

    print_listing(hosts)

    if args.test:
        workers = args.test_workers or env_test_threads()
        run_tests(hosts, args.test_timeout, args.num_predict, workers)
        # Persistir los resultados de las pruebas en la caché.
        save_cache(hosts)
        print(f"\n[*] Resultados de las pruebas guardados en {CACHE_FILE}")

    if args.chat:
        start_chat(hosts, args.timeout)


if __name__ == "__main__":
    main()
