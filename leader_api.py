import os
import json
import time
import threading
import subprocess
from hashlib import sha256
from uuid import uuid4

from flask import Flask, request, jsonify
from make_embeddings import gerar_embedding

app = Flask(__name__)

# ------------------ CONFIG ------------------ #

TOPIC_UPDATES = "vector-updates3"     # MESMO TÓPICO DOS PEERS
VECTOR_FILE = "vetor_documentos.json"
PENDING_FILE = "vetor_temp.json"

HEARTBEAT_INTERVAL = 5  # segundos entre heartbeats

LEADER_IPFS_ID = None

# Estruturas pendentes (RNF3)
pending_versions = {}
lock = threading.Lock()

# ------------------ RF2: Tarefas de pesquisa ------------------ #

tasks = {}
tasks_cond = threading.Condition()

# ------------------ IPFS HELPERS ------------------ #


def obter_ipfs_id_lider():
    global LEADER_IPFS_ID
    if LEADER_IPFS_ID is not None:
        return LEADER_IPFS_ID

    proc = subprocess.run(
        ["ipfs", "id", "-f", "<id>"],
        capture_output=True,
        text=True
    )
    LEADER_IPFS_ID = proc.stdout.strip()
    print(f"[LÍDER] IPFS ID: {LEADER_IPFS_ID}")
    return LEADER_IPFS_ID


def get_real_peers():
    """
    Descobre peers ligados ao PubSub (exceto o próprio líder).
    """
    try:
        proc = subprocess.run(
            ["ipfs", "pubsub", "peers", TOPIC_UPDATES],
            capture_output=True,
            text=True
        )
        peers = proc.stdout.strip().split("\n")
        peers = [p for p in peers if p]

        leader_id = obter_ipfs_id_lider()
        peers = [p for p in peers if p != leader_id]

        return peers
    except Exception as e:
        print("[LÍDER] Erro a obter peers:", e)
        return []

# ------------------ PERSISTÊNCIA ------------------ #


def load_vector():
    if not os.path.exists(VECTOR_FILE):
        return {"versoes": []}
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vector(v):
    with open(VECTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


def save_pending(v):
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


def hash_vector(v):
    return sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()

# ------------------ PUBSUB ------------------ #


def pubsub_publish(message_dict: dict):
    msg_json = json.dumps(message_dict)

    proc = subprocess.Popen(
        ["ipfs", "pubsub", "pub", TOPIC_UPDATES],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    proc.communicate(msg_json + "\n")

    if proc.returncode != 0:
        print("[LÍDER] Erro ao publicar no PubSub")

# ------------------ HEARTBEAT (compatível com RAFT) ------------------ #


def heartbeat_loop():

    while True:
        time.sleep(HEARTBEAT_INTERVAL)

        vetor = load_vector()
        ultimo_commit = len(vetor.get("versoes", []))

        msg = {
            "tipo": "heartbeat",
            "leader_id": "leader_api",
            "term": 0,
            "timestamp": time.time(),
            "ultimo_commit": ultimo_commit
        }

        pubsub_publish(msg)

# ------------------ COMMIT (RNF3) ------------------ #


def commit_version(versao: int, info: dict):
    vetor_pendente = info["vector"]
    h = info["hash"]

    vetor_confirmado = load_vector()
    if "versoes" not in vetor_confirmado:
        vetor_confirmado["versoes"] = []

    nova_entry = vetor_pendente["versoes"][0]
    vetor_confirmado["versoes"].append(nova_entry)

    save_vector(vetor_confirmado)
    save_pending(vetor_pendente)

    print(f"[LÍDER] Versão {versao} confirmada.")

    msg = {
        "tipo": "commit",
        "versao": versao,
        "hash": h
    }
    pubsub_publish(msg)
    print(f"[LÍDER] COMMIT enviado para versão {versao}.")

# ------------------ ACKS (RNF3) ------------------ #


def process_update_ack(payload: dict):
    versao = payload.get("versao")
    h = payload.get("hash")
    peer_id = payload.get("peer_id", "peer_desconhecido")

    with lock:
        info = pending_versions.get(versao)
        if info is None:
            return

        if h != info["hash"]:
            return

        info["acks"].add(peer_id)
        num_acks = len(info["acks"])

        real_peers = get_real_peers()
        cluster_size = len(real_peers)
        majority = (cluster_size // 2) + 1

        print(f"[LÍDER] ACK de {peer_id}: {num_acks}/{majority}")

        if num_acks >= majority:
            pending_versions.pop(versao, None)
            commit_version(versao, info)

# ------------------ RF2: callbacks PubSub ------------------ #


def register_task_claim(payload: dict):
    """
    Recebemos info de que um peer aceitou processar uma tarefa.
    """
    task_id = payload.get("task_id")
    peer_id = payload.get("peer_id")
    if not task_id or not peer_id:
        return

    with tasks_cond:
        tinfo = tasks.setdefault(task_id, {})
        # só registamos o primeiro peer que reclamar
        if tinfo.get("peer_id") is None:
            tinfo["peer_id"] = peer_id
            tinfo["status"] = "assigned"
            print(f"[LÍDER] Tarefa {task_id} atribuída a {peer_id}")
        tasks_cond.notify_all()


def register_task_output_response(payload: dict):
    """
    Recebemos a resposta de uma tarefa de pesquisa.
    """
    task_id = payload.get("task_id")
    ready = payload.get("ready", False)
    if not task_id:
        return

    with tasks_cond:
        tinfo = tasks.setdefault(task_id, {})
        if ready:
            tinfo["output"] = payload.get("output")
            tinfo["status"] = "done"
            tinfo["peer_id"] = payload.get("peer_id", tinfo.get("peer_id"))
            print(f"[LÍDER] Output recebido para tarefa {task_id}")
        else:
            # apenas marca como ainda em processamento
            if tinfo.get("status") != "done":
                tinfo["status"] = "processing"
        tasks_cond.notify_all()

# ------------------ LISTENER PUBSUB ------------------ #


def process_pubsub_message(payload: dict):
    tipo = payload.get("tipo")

    if tipo == "ack":
        process_update_ack(payload)
    elif tipo == "task_claim":
        register_task_claim(payload)
    elif tipo == "task_output_response":
        register_task_output_response(payload)


def listen_pubsub_for_acks():
    print(f"[LÍDER] A escutar ACKs e RF2 no tópico '{TOPIC_UPDATES}'...")

    while True:
        try:
            proc = subprocess.Popen(
                ["ipfs", "pubsub", "sub", TOPIC_UPDATES],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            msg = ""
            nivel = 0

            while True:
                c = proc.stdout.read(1)
                if not c:
                    time.sleep(0.01)
                    continue

                if c == "{":
                    nivel += 1

                if nivel > 0:
                    msg += c

                if c == "}":
                    nivel -= 1
                    if nivel == 0:
                        try:
                            payload = json.loads(msg)
                            process_pubsub_message(payload)
                        except Exception as e:
                            print("[LÍDER] ERRO JSON:", e)
                        msg = ""

        except Exception as e:
            print("[LÍDER] Erro no listener de ACKs/RF2:", e)
            time.sleep(3)

# ------------------ RF2: ENDPOINTS HTTP ------------------ #


def wait_for_output(task_id: str, timeout: float = 5.0):
    """
    Espera até 'timeout' segundos por output de uma tarefa.
    """
    deadline = time.time() + timeout
    with tasks_cond:
        while True:
            tinfo = tasks.get(task_id)
            if tinfo and tinfo.get("output") is not None:
                return tinfo

            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            tasks_cond.wait(timeout=remaining)


@app.route("/search", methods=["POST"])
def search():
    """
    Fase 1 RF2:
    - recebe prompt
    - gera task_id
    - envia pedido de processamento para a rede
    - devolve task_id ao cliente
    """
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"erro": "Campo 'prompt' em falta"}), 400

    task_id = str(uuid4())

    with tasks_cond:
        tasks[task_id] = {
            "prompt": prompt,
            "peer_id": None,
            "status": "pending",
            "output": None
        }

    msg = {
        "tipo": "task_request",
        "task_id": task_id,
        "prompt": prompt,
        "timestamp": time.time()
    }
    pubsub_publish(msg)
    print(f"[LÍDER] RF2: task_request enviado ({task_id})")

    return jsonify({"task_id": task_id}), 200


@app.route("/getOutput/<task_id>", methods=["GET"])
def get_output(task_id):
    """
    - cliente pede a resposta para um task_id
    - líder pergunta ao peer responsável
    - se houver output devolve, senão diz que está em processamento
    """
    with tasks_cond:
        tinfo = tasks.get(task_id)

        if tinfo is None:
            return jsonify({"erro": "task_id desconhecido"}), 404

        # se já temos output em cache, devolvemos logo
        if tinfo.get("output") is not None:
            return jsonify({
                "task_id": task_id,
                "status": "done",
                "output": tinfo["output"],
                "peer_id": tinfo.get("peer_id")
            }), 200

        peer_id = tinfo.get("peer_id")

    if not peer_id:
        # ainda ninguém reclamou a tarefa
        return jsonify({
            "task_id": task_id,
            "status": "pending_assignment"
        }), 202

    # pedir explicitamente o output ao peer responsável
    msg = {
        "tipo": "task_output_request",
        "task_id": task_id,
        "peer_id": peer_id
    }
    pubsub_publish(msg)
    print(f"[LÍDER] RF2: task_output_request para {peer_id} ({task_id})")

    tinfo = wait_for_output(task_id, timeout=5.0)
    if tinfo is None or tinfo.get("output") is None:
        return jsonify({
            "task_id": task_id,
            "status": "processing"
        }), 202

    return jsonify({
        "task_id": task_id,
        "status": "done",
        "output": tinfo["output"],
        "peer_id": tinfo.get("peer_id")
    }), 200

# ------------------ /add_file (RNF3) ------------------ #


@app.route("/add_file", methods=["POST"])
def add_file():
    if "file" not in request.files:
        return jsonify({"erro": "Ficheiro não encontrado"}), 400

    f = request.files["file"]
    filename = f.filename or "ficheiro"
    temp_path = f"temp_{int(time.time())}_{filename}"
    f.save(temp_path)

    # ADD IPFS
    proc = subprocess.run(
        ["ipfs", "add", "-Q", "--pin=true", temp_path],
        text=True,
        capture_output=True
    )
    if proc.returncode != 0:
        os.remove(temp_path)
        return jsonify({"erro": "Falha IPFS"}), 500

    cid = proc.stdout.strip()
    print(f"[LÍDER] Ficheiro {filename} → CID = {cid}")

    embedding = gerar_embedding(temp_path)
    os.remove(temp_path)

    vetor_atual = load_vector()
    versoes_atual = vetor_atual.get("versoes", [])
    nova_versao_num = len(versoes_atual) + 1

    nova_entry = {
        "versao": nova_versao_num,
        "cid": cid,
        "embedding": embedding
    }

    vetor_pendente = {"versoes": [nova_entry]}
    save_pending(vetor_pendente)

    h = hash_vector(vetor_pendente)

    with lock:
        pending_versions[nova_versao_num] = {
            "vector": vetor_pendente,
            "hash": h,
            "acks": set()
        }

    msg = {
        "tipo": "update_request",
        "versao": nova_versao_num,
        "cid": cid,
        "embedding": embedding,
        "hash": h
    }
    pubsub_publish(msg)
    print(f"[LÍDER] update_request enviado.")

    return jsonify({
        "mensagem": "Ficheiro recebido e update enviado.",
        "cid": cid,
        "versao": nova_versao_num,
        "hash": h
    }), 200

# ------------------ MAIN ------------------ #


if __name__ == "__main__":
    threading.Thread(target=listen_pubsub_for_acks, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()

    print("Líder HTTP em http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
