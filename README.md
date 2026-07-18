# AI Scanner — Ollama

Herramienta para localizar servidores de IA expuestos (sin autenticación) con
Shodan y trabajar con ellos. Soporta **Ollama** y servidores compatibles con
la API de **OpenAI** (llama.cpp, vLLM, LocalAI, LM Studio, text-gen-webui).
Incluye:

- **[find_ollama.py](find_ollama.py)** — scanner de línea de comandos (búsqueda,
  pruebas de velocidad y chat).
- **[tui.py](tui.py)** — interfaz de terminal (TUI) para hacer todo lo anterior
  de forma visual e interactiva.
- Una guía de comandos `curl` para la API HTTP de Ollama.

> ⚠️ **Uso responsable:** interactúa únicamente con servidores propios o para
> los que tengas autorización. Los servidores encontrados con Shodan están
> expuestos por mala configuración; consultarlos o usarlos sin permiso puede
> ser ilegal.

---

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install shodan requests python-dotenv textual
```

Crea un fichero `.env` en la raíz con tu clave de Shodan y, opcionalmente, el
número de hilos para las pruebas de modelos:

```
SHODAN_API_KEY=tu_clave_aqui
TEST_THREADS=20              # hilos para --test / tecla t en la TUI (def: 10)
```

Las pruebas de modelos (`--test` en la CLI y la tecla `t` en la TUI) se ejecutan
en paralelo con ese número de hilos, así que terminan mucho antes. En la CLI se
puede sobrescribir con `--test-workers N`.

---

## Scanner por línea de comandos (find_ollama.py)

```bash
source venv/bin/activate
python find_ollama.py                 # lista desde la caché (o busca si no existe)
python find_ollama.py --update        # fuerza nueva búsqueda en Shodan
python find_ollama.py --limit 20      # nº máximo de resultados a procesar
python find_ollama.py --no-probe      # solo datos de Shodan, sin contactar hosts
python find_ollama.py --test          # prueba cada modelo y mide su velocidad
python find_ollama.py --chat          # elige modelo y servidor e inicia un chat
```

| Opción            | Descripción                                                     |
| ----------------- | --------------------------------------------------------------- |
| `--update`        | Vuelve a consultar Shodan y sobrescribe la caché.               |
| `--limit N`       | Máximo de resultados a procesar (def: 100).                     |
| `--no-probe`      | No contacta los hosts; usa solo el banner de Shodan.            |
| `--chat`          | Flujo interactivo: eliges **modelo** y luego **servidor**.      |
| `--test`          | Prueba cada `(servidor, modelo)` y reporta velocidad (tok/s).   |
| `--test-timeout`  | Timeout por prueba en `--test` (def: 60 s).                     |
| `--num-predict`   | Tokens a generar en cada prueba (def: 16).                      |
| `--test-workers`  | Pruebas en paralelo en `--test` (def: 10).                      |
| `--providers`     | Proveedores a buscar, separados por comas, o `all` (def: all).  |

Los resultados se cachean en **`ollama_hosts.json`**. Cualquier ejecución
reutiliza esa caché; solo `--update` (o su ausencia inicial) consulta Shodan.
Los datos de `--test` también se guardan, así que los colores de estado se
mantienen entre ejecuciones.

### Proveedores soportados

Además de Ollama, busca otros servidores de IA que suelen exponerse **sin
autenticación**. Los que hablan la API compatible con OpenAI comparten los
endpoints `/v1/models` y `/v1/chat/completions`:

| Proveedor    | Protocolo | Puerto | Notas                                       |
| ------------ | --------- | ------ | ------------------------------------------- |
| `ollama`     | Ollama    | 11434  | API nativa (`/api/tags`, `/api/chat`).      |
| `llamacpp`   | OpenAI    | 8080   | `llama-server`; fingerprint fiable.         |
| `vllm`       | OpenAI    | 8000   | Fingerprint aproximado (se confirma al sondear). |
| `localai`    | OpenAI    | 8080   | Cabecera `Server: LocalAI`.                 |
| `lmstudio`   | OpenAI    | 1234   | Fingerprint débil.                          |
| `tgwebui`    | OpenAI    | 5000   | text-generation-webui (oobabooga).          |

```bash
python find_ollama.py --update --providers ollama,vllm,llamacpp   # solo estos
python find_ollama.py --update --providers all                     # todos (def)
```

> Las consultas de Shodan de algunos proveedores son aproximadas; la
> confirmación real de que es un servidor válido y su lista de modelos se hace
> al sondear el host en vivo. Puedes afinar las consultas editando el
> diccionario `PROVIDERS` en [find_ollama.py](find_ollama.py).

---

## Interfaz TUI (tui.py)

Una interfaz de terminal que reúne todas las funciones (buscar, probar, elegir
y chatear) en una sola pantalla, navegable con **teclado y ratón**.

```bash
source venv/bin/activate
python tui.py
```

### Distribución de la pantalla

```
┌ MODELOS ─────────────┬─────────────────────────────────┐
│ [filtro de modelos]  │  estado (modelo/servidor activo) │
│ llama3.2  (5 srv)    │                                  │
│ qwen3.6   (3 srv)    │  registro de chat                │
│ ...                  │  (respuestas con Markdown)       │
├ SERVIDORES ──────────┤                                  │
│ 1.2.3.4  50 tok/s    │  [respuesta en streaming]        │
│ 5.6.7.8  ✗ FALLO     │  > escribe tu mensaje…           │
└──────────────────────┴──────────────────────────────────┘
```

- **MODELOS**: todos los modelos agregados, ordenados por disponibilidad, con
  una caja de **filtro** por texto. Un modelo se pinta en **verde** si al menos
  un servidor que lo ofrece ha pasado la prueba.
- **SERVIDORES**: al elegir un modelo, muestra los servidores que lo ofrecen,
  **ordenados de más rápido a más lento**. En **verde** con `tok/s · latencia`
  los que funcionan, en **rojo** los que fallaron la prueba.
- **Chat**: registro con las respuestas renderizadas como **Markdown** (listas,
  tablas, código…) y streaming token a token.

### Flujo de trabajo

1. Pulsa **`u`** para buscar/actualizar servidores en Shodan (o se carga la caché).
2. Pulsa **`t`** para probar todos los modelos y marcar en verde los que funcionan.
3. Elige un **modelo** (`↑↓` + `Enter`) y luego un **servidor** (`↑↓` + `Enter`).
4. Escribe abajo y pulsa `Enter` para chatear.

### Atajos de teclado

| Tecla        | Acción                                                        |
| ------------ | ------------------------------------------------------------- |
| `↑` / `↓`    | Moverse dentro de la lista enfocada                           |
| `Enter`      | Elegir modelo (→ servidores) / servidor (→ escribir)          |
| `←` / `→`    | Saltar entre las listas de modelos y servidores              |
| `Esc`        | Salir de la caja de texto y volver a navegar                  |
| `Tab`        | Cambiar de panel                                             |
| `F2` / `F3`  | Ir al filtro de modelos / a la caja de escribir             |
| `u`          | Actualizar la lista desde Shodan                             |
| `t`          | Probar los modelos y medir su velocidad                     |
| `o`          | Mostrar **solo disponibles** (verde) / todos                |
| `r`          | Reiniciar el contexto del chat                              |
| `Ctrl+Y`     | Copiar la última respuesta del chat al portapapeles         |
| `Ctrl+B`     | Copiar el modelo o servidor resaltado (o el par activo)     |
| `Ctrl+L`     | Limpiar el registro                                         |
| `q`          | Salir                                                        |

> Las teclas de una sola letra (`u`, `t`, `o`, `r`, `q`) actúan cuando el foco
> **no** está en una caja de texto. Si estás escribiendo, pulsa `Esc` primero.
> Los atajos `Ctrl+…` funcionan también mientras escribes.

**Copiar al portapapeles** usa OSC 52, soportado por la mayoría de terminales
modernos (kitty, wezterm, foot, alacritty, tmux con `set-clipboard on`…). Para
seleccionar texto con el ratón, mantén pulsado `Shift` mientras arrastras (así
saltas la captura de ratón de la TUI y usas la selección nativa del terminal).

---

## Comandos básicos con curl

La API de Ollama escucha por defecto en el puerto **11434**. En los ejemplos
siguientes se usa una variable `HOST` para apuntar al servidor:

```bash
export HOST=http://localhost:11434     # servidor local
# export HOST=http://IP:11434          # servidor remoto
```

### 1. Comprobar que el servidor está vivo

```bash
curl $HOST
# -> "Ollama is running"
```

### 2. Listar los modelos disponibles (instalados)

```bash
curl $HOST/api/tags
```

Con formato legible (si tienes `jq`):

```bash
curl -s $HOST/api/tags | jq '.models[].name'
```

### 3. Listar los modelos cargados en memoria ahora mismo

```bash
curl $HOST/api/ps
```

### 4. Ver información / detalles de un modelo

```bash
curl $HOST/api/show -d '{
  "model": "llama3.2"
}'
```

### 5. Generar una respuesta (prompt único)

Respuesta en streaming (por defecto):

```bash
curl $HOST/api/generate -d '{
  "model": "llama3.2",
  "prompt": "¿Por qué el cielo es azul?"
}'
```

Respuesta completa de una sola vez (sin streaming):

```bash
curl $HOST/api/generate -d '{
  "model": "llama3.2",
  "prompt": "¿Por qué el cielo es azul?",
  "stream": false
}'
```

Extraer solo el texto con `jq`:

```bash
curl -s $HOST/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Dame un dato curioso",
  "stream": false
}' | jq -r '.response'
```

### 6. Chat conversacional (con roles y contexto)

```bash
curl $HOST/api/chat -d '{
  "model": "llama3.2",
  "stream": false,
  "messages": [
    { "role": "system",  "content": "Eres un asistente conciso." },
    { "role": "user",    "content": "Hola, ¿quién eres?" }
  ]
}'
```

### 7. Generar embeddings

```bash
curl $HOST/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "Texto a vectorizar"
}'
```

### 8. Gestión de modelos

Descargar (pull) un modelo:

```bash
curl $HOST/api/pull -d '{ "model": "llama3.2" }'
```

Borrar un modelo:

```bash
curl -X DELETE $HOST/api/delete -d '{ "model": "llama3.2" }'
```

Copiar un modelo:

```bash
curl $HOST/api/copy -d '{ "source": "llama3.2", "destination": "mi-llama" }'
```

### 9. Ver la versión del servidor

```bash
curl $HOST/api/version
```

---

## Parámetros útiles en generate / chat

Se pasan dentro del objeto `options`:

```bash
curl $HOST/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Escribe un haiku",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_predict": 128,
    "seed": 42
  }
}'
```

| Parámetro      | Descripción                                         |
| -------------- | --------------------------------------------------- |
| `temperature`  | Creatividad (0 = determinista, 1+ = más aleatorio)  |
| `top_p`        | Muestreo por núcleo (nucleus sampling)              |
| `num_predict`  | Máximo de tokens a generar                          |
| `seed`         | Semilla para resultados reproducibles               |
| `stop`         | Lista de secuencias que detienen la generación      |

---

## Referencia

- Documentación oficial de la API: <https://github.com/ollama/ollama/blob/main/docs/api.md>
