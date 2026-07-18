#!/usr/bin/env python3
"""Busca servidores Ollama expuestos usando la API de Shodan y lista, por
cada IP, los modelos disponibles.

Uso:
    python find_ollama.py [--limit N] [--no-probe] [--timeout S]
    python find_ollama.py --update          # refresca la caché desde Shodan
    python find_ollama.py --chat            # chatea usando la caché existente

Los resultados se guardan en ollama_hosts.json. En modo --chat (o cualquier
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

# Query de Shodan: Ollama escucha por defecto en el puerto 11434 y su
# servidor HTTP responde con la cabecera "Server: ollama".
SHODAN_QUERY = 'product:Ollama port:11434'

# Fichero donde se cachean los resultados de la búsqueda.
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ollama_hosts.json")


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


def get_models_from_host(ip: str, port: int, timeout: float):
    """Consulta directamente /api/tags del host para obtener la lista de
    modelos. Devuelve una lista de nombres o None si no responde."""
    url = f"http://{ip}:{port}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "?") for m in data.get("models", [])]
    except Exception:
        return None


def models_from_shodan_banner(match: dict):
    """Intenta extraer los modelos que Shodan ya haya capturado en el banner
    (algunos hosts exponen /api/tags en los datos indexados)."""
    data = match.get("data", "") or ""
    models = []
    if '"models"' in data and '"name"' in data:
        import re
        for name in re.findall(r'"name"\s*:\s*"([^"]+)"', data):
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


def chat_stream_iter(host_url: str, model: str, messages: list, timeout: float):
    """Generador que envía la conversación a /api/chat en streaming y va
    cediendo (yield) cada fragmento de texto del asistente. Si hay un error,
    cede una tupla ("__error__", mensaje). Reutilizable por CLI y TUI."""
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


def chat_stream(host_url: str, model: str, messages: list, timeout: float):
    """Versión CLI: imprime la respuesta en streaming. Devuelve el texto
    completo del asistente o None si falla."""
    full = []
    try:
        for piece in chat_stream_iter(host_url, model, messages, timeout):
            if isinstance(piece, tuple):  # ("__error__", msg)
                print(f"    [{piece[1]}]")
                return None
            full.append(piece)
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        print("\n    [respuesta interrumpida]")
    print()
    return "".join(full)


def test_model(ip: str, port: int, model: str, timeout: float, num_predict: int):
    """Envía una generación mínima a un modelo y mide su rendimiento.

    Devuelve un dict con:
      ok            -> True/False
      latency       -> tiempo total de la petición (s)
      tokens_per_sec-> velocidad de generación según métricas de Ollama
      eval_count    -> nº de tokens generados
      error         -> descripción si ok=False
    """
    url = f"http://{ip}:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": "Responde solo con la palabra: OK",
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout", "latency": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "latency": time.perf_counter() - t0}

    latency = time.perf_counter() - t0
    if isinstance(data, dict) and data.get("error"):
        return {"ok": False, "error": str(data["error"])[:60], "latency": latency}

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
        res = test_model(h["ip"], h["port"], m, timeout, num_predict)
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
        print(f"\nModelos que fallan ({len(ko)}):")
        for r in ko:
            print(f"  - {r['ip']}:{r['port']}  {r['model']}  ({r['error']})")

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
    # Timeout más generoso para la generación que para el sondeo inicial.
    chat_timeout = max(default_timeout, 120.0)

    print("\n" + "=" * 70)
    print(f"Chat con {model} @ {host['ip']}:{host['port']}")
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
        reply = chat_stream(host_url, model, messages, chat_timeout)
        if reply is None:
            # Deshacer el último mensaje para poder reintentar.
            messages.pop()
        else:
            messages.append({"role": "assistant", "content": reply})

    print("\nSesión de chat finalizada.")


def fetch_from_shodan(limit: int, no_probe: bool, timeout: float):
    """Consulta Shodan, sondea los hosts en vivo y devuelve la lista de hosts."""
    load_dotenv()
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        sys.exit("ERROR: no se encontró SHODAN_API_KEY en el entorno ni en .env")

    api = shodan.Shodan(api_key)

    print(f"[*] Buscando servidores Ollama en Shodan: {SHODAN_QUERY!r}")
    try:
        results = api.search(SHODAN_QUERY, limit=limit)
    except shodan.APIError as e:
        sys.exit(f"ERROR de Shodan: {e}")

    total = results.get("total", 0)
    matches = results.get("matches", [])
    print(f"[*] Total en Shodan: {total}. Procesando {len(matches)} resultados.\n")

    hosts = []
    for m in matches:
        hosts.append({
            "ip": m.get("ip_str"),
            "port": m.get("port", 11434),
            "org": m.get("org", ""),
            "country": (m.get("location", {}) or {}).get("country_name", ""),
            "shodan_models": models_from_shodan_banner(m),
        })

    # Consultar modelos en vivo (en paralelo) salvo que se pida --no-probe
    if not no_probe:
        def probe(h):
            h["live_models"] = get_models_from_host(h["ip"], h["port"], timeout)
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
        print(f"\nIP: {h['ip']}:{h['port']}  [{estado}]")
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
    parser = argparse.ArgumentParser(description="Busca servidores Ollama con Shodan")
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
    parser.add_argument("--test-workers", type=int, default=10,
                        help="Pruebas en paralelo en --test (def: 10)")
    args = parser.parse_args()

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
        hosts = fetch_from_shodan(args.limit, args.no_probe, args.timeout)
        save_cache(hosts)
        print(f"[*] Resultados guardados en {CACHE_FILE}")
    else:
        hosts = cached
        print(f"[*] Usando caché de {CACHE_FILE} ({len(hosts)} servidores). "
              f"Usa --update para refrescar.")

    print_listing(hosts)

    if args.test:
        run_tests(hosts, args.test_timeout, args.num_predict, args.test_workers)
        # Persistir los resultados de las pruebas en la caché.
        save_cache(hosts)
        print(f"\n[*] Resultados de las pruebas guardados en {CACHE_FILE}")

    if args.chat:
        start_chat(hosts, args.timeout)


if __name__ == "__main__":
    main()
