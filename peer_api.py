import json
import threading
import time
import random
import subprocess
import os
from hashlib import sha256

import numpy as np
import faiss

from flask import Flask, request, jsonify
from make_embeddings import gerar_embedding

app = Flask(__name__)


TOPIC_UPDATES = "vector-updates3"

VECTOR_FILE = "vetor_documentos.json"      # estado permanente
TEMP_FILE = "vetor_temp.json"              # follower: update pendente
PENDING_FILE = "vetor_temp_lider.json"     # líder: update pendente

NODE_ID = None
HTTP_PORT = None
CLUSTER_SIZE = None        # nº de peers RAFT (ex: 3)

lock = threading.Lock()

# ------------------ IPFS ------------------ #

PEER_IPFS_ID = None


def obter_ipfs_id_peer():
    global PEER_IPFS_ID
    if PEER_IPFS_ID is not None:
        return PEER_IPFS_ID
    proc = subprocess.run(
        ["ipfs", "id", "-f", "<id>"],
        text=True,
        capture_output=True
    )
    PEER_IPFS_ID = proc.stdout.strip()
    print(f"[{NODE_ID}] IPFS ID = {PEER_IPFS_ID}")
    return PEER_IPFS_ID


def get_real_peers():
    """
    Só para debug. Para RAFT usamos CLUSTER_SIZE.
    """
    try:
        proc = subprocess.run(
            ["ipfs", "pubsub", "peers", TOPIC_UPDATES],
            text=True,
            capture_output=True
        )
        peers = [p for p in proc.stdout.strip().split("\n") if p]
        my_id = obter_ipfs_id_peer()
        peers = [p for p in peers if p != my_id]
        return peers
    except Exception as e:
        print(f"[{NODE_ID}] ERRO descoberta dinâmica: {e}")
        return []

def load_vector():
    if not os.path.exists(VECTOR_FILE):
        return {"versoes": []}
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vector(v):
    with open(VECTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


def load_temp():
    if not os.path.exists(TEMP_FILE):
        return None
    with open(TEMP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_temp(v):
    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


def load_pending():
    if not os.path.exists(PENDING_FILE):
        return None
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pending(v):
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


def hash_vector(v):
    return sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def get_local_last_version():
    vetor = load_vector()
    return len(vetor.get("versoes", []))

# ------------------ ESTADO RAFT ------------------ #

state = "follower"     # follower, candidate, leader
current_term = 0
voted_for = None
last_heartbeat = time.time()
leader_dead = False

ELECTION_RANGE = (2.0, 4.0)


def new_timeout():
    return time.time() + random.uniform(*ELECTION_RANGE)


next_election = new_timeout()

HEARTBEAT_INTERVAL = 10

# Versão máxima conhecida no cluster
cluster_max_version = 0

# Votos por candidato (para consenso)
# term -> {candidato: set(peer_ids)}
votos_por_candidato = {}
votos_recebidos = set()

# Versões pendentes (quando ESTE peer é líder)
# versao -> {"vector":..., "hash":..., "acks": set()}
pending_versions = {}


SYNC_REQUEST_INTERVAL = 5.0   # segundos entre tentativas de sync
last_sync_request = 0.0       # para não spammar pedidos de sync


def append_entries(entries):
    """
    Adiciona ao vetor_documentos as entradas ainda não existentes.
    Assume que 'entries' vem ordenado por versao.
    """
    vetor = load_vector()
    if "versoes" not in vetor:
        vetor["versoes"] = []

    local_last = len(vetor["versoes"])

    for e in entries:
        v_num = e.get("versao")
        if v_num is None:
            continue
        # se já tenho esta ou uma versão mais avançada, ignoro
        if v_num <= local_last:
            continue
        vetor["versoes"].append(e)
        local_last = v_num

    save_vector(vetor)
    return local_last


def need_sync():
    """
    True se este nó estiver com o log desatualizado face ao cluster.
    """
    local_version = get_local_last_version()
    return local_version < cluster_max_version


def request_sync(force=False):
    """
    Envia um pedido de sincronização ao líder atual.
    """
    global last_sync_request

    if state == "leader":
        return  # o líder é a fonte de verdade, não pede sync

    now = time.time()
    if not force and (now - last_sync_request) < SYNC_REQUEST_INTERVAL:
        return

    last_sync_request = now

    msg = {
        "tipo": "sync_request",
        "peer_id": NODE_ID,
        "from_version": get_local_last_version(),
        "term": current_term
    }
    print(
        f"[{NODE_ID}] Pedido de sync: from_version={msg['from_version']}, "
        f"cluster_max={cluster_max_version}"
    )
    pubsub_publish(msg)


def handle_sync_request(payload: dict):
    """
    LÍDER:
      - recebe pedido de sync de um follower
      - devolve as entradas em falta (versao > from_version)
    """
    if state != "leader":
        return

    from_version = payload.get("from_version", 0)
    dest_peer = payload.get("peer_id")
    if dest_peer is None:
        return

    vetor = load_vector()
    versoes = vetor.get("versoes", [])
    entries = [e for e in versoes if e.get("versao", 0) > from_version]

    if not entries:
        return

    msg = {
        "tipo": "sync_response",
        "dest_peer": dest_peer,
        "entries": entries,
        "term": current_term
    }
    print(
        f"[{NODE_ID}] Enviar sync_response para {dest_peer} "
        f"({len(entries)} entradas desde v>{from_version})"
    )
    pubsub_publish(msg)


def handle_sync_response(payload: dict):
    """
    FOLLOWER:
      - aplica entradas recebidas do líder
    """
    global cluster_max_version

    dest = payload.get("dest_peer")
    if dest != NODE_ID:
        return  # não é para mim

    entries = payload.get("entries", [])
    if not entries:
        return

    last_after = append_entries(entries)
    if last_after > cluster_max_version:
        cluster_max_version = last_after

    print(f"[{NODE_ID}] Sync aplicado até versão {last_after}")

# ------------------ PUBSUB ------------------ #

def pubsub_publish(message: dict):
    msg = json.dumps(message)
    try:
        subprocess.Popen(
            ["ipfs", "pubsub", "pub", TOPIC_UPDATES],
            stdin=subprocess.PIPE,
            text=True
        ).communicate(msg + "\n")
    except Exception as e:
        print(f"[{NODE_ID}] ERRO ao publicar PubSub: {e}")

# ------------------ HEARTBEATS ------------------ #


def receber_heartbeat(payload: dict):
    """
    Recebe heartbeats tanto do leader_api (leader_id='leader_api', term=0)
    como de um peer líder (leader_id = NODE_ID, term > 0).
    """
    global last_heartbeat, state, leader_dead, current_term, cluster_max_version

    leader_id = payload.get("leader_id")
    term = payload.get("term", 0)

    # Ignorar os nossos próprios heartbeats
    if leader_id == NODE_ID:
        last_heartbeat = time.time()
        return

    last_heartbeat = time.time()
    leader_dead = False

    if term >= current_term:
        current_term = term
        state = "follower"

    hb_version = payload.get("ultimo_commit")

    # Atualizar cluster_max_version se líder tiver log mais avançado
    if hb_version is not None and hb_version > cluster_max_version:
        cluster_max_version = hb_version

    # Se eu estiver atrás, peço sync
    if hb_version is not None and need_sync():
        request_sync()


def heartbeat_loop():
    """
    Se este peer for líder, envia heartbeats periódicos.
    """
    global cluster_max_version

    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        if state != "leader":
            continue

        ultimo_commit = get_local_last_version()
        if ultimo_commit > cluster_max_version:
            cluster_max_version = ultimo_commit

        msg = {
            "tipo": "heartbeat",
            "leader_id": NODE_ID,
            "term": current_term,
            "timestamp": time.time(),
            "ultimo_commit": ultimo_commit
        }
        pubsub_publish(msg)

# ------------------ RAFT: ELEIÇÕES ------------------ #

def receber_pedido_voto(payload: dict):
    """
    Outro peer pede o nosso voto.
    Só votamos em candidatos com log pelo menos tão atualizado como o nosso.
    """
    global voted_for, current_term, state

    candidato = payload.get("candidato")
    term = payload.get("term")
    cand_last_version = payload.get("last_version", 0)

    if term is None or candidato is None:
        return

    if term < current_term:
        return

    my_last_version = get_local_last_version()

    # Regra RAFT: não voto em candidato mais desatualizado que eu
    if cand_last_version < my_last_version:
        print(
            f"[{NODE_ID}] Recuso voto a {candidato}: "
            f"cand={cand_last_version} < my={my_last_version}"
        )
        return

    if term > current_term:
        current_term = term
        voted_for = None
        state = "follower"

    if voted_for is None or voted_for == candidato:
        voted_for = candidato
        msg = {
            "tipo": "voto",
            "peer_id": NODE_ID,
            "term": current_term,
            "candidato": candidato
        }
        pubsub_publish(msg)


def receber_voto(payload: dict):
    """
    Recebemos um voto (para qualquer candidato).
    Atualizamos tabela de consenso e, se formos candidato, verificamos maioria.
    """
    global votos_recebidos, votos_por_candidato, state

    peer_id = payload.get("peer_id")
    term = payload.get("term")
    candidato = payload.get("candidato")

    if peer_id is None or term is None or candidato is None:
        return

    # tabela global
    if term not in votos_por_candidato:
        votos_por_candidato[term] = {}
    if candidato not in votos_por_candidato[term]:
        votos_por_candidato[term][candidato] = set()
    votos_por_candidato[term][candidato].add(peer_id)

    majority = (CLUSTER_SIZE // 2) + 1

    print(
        f"[{NODE_ID}] Tabela votos (term={term}): "
        f"{ {cand: list(vs) for cand, vs in votos_por_candidato[term].items()} }"
    )

    if state != "candidate":
        return

    if candidato != NODE_ID or term != current_term:
        return

    votos_recebidos.add(peer_id)
    print(f"[{NODE_ID}] Recebi voto de {peer_id}: {len(votos_recebidos)}/{majority}")

    if len(votos_recebidos) >= majority:
        tornar_lider()


def tornar_lider():
    """
    Atingimos maioria de votos.
    """
    global state, leader_dead

    state = "leader"
    leader_dead = False
    print(f"[{NODE_ID}] >>> ELEITO LÍDER! (term={current_term})")

    recover_pending_state()


def iniciar_eleicao():
    """
    Inicia uma eleição RAFT.
    Só podemos ser candidato se estivermos atualizados com o cluster.
    """
    global state, current_term, voted_for, votos_recebidos, next_election

    local_version = get_local_last_version()
    if local_version < cluster_max_version:
        print(
            f"[{NODE_ID}] NÃO posso ser candidato (desatualizado). "
            f"local={local_version}, cluster={cluster_max_version}"
        )

        # tentar obter sync antes da próxima tentativa

        request_sync(force=True)
        return

    state = "candidate"
    current_term += 1
    voted_for = NODE_ID
    votos_recebidos = {NODE_ID}

    print(
        f"[{NODE_ID}] Iniciar eleição (term={current_term}, "
        f"last_version={local_version})"
    )

    msg = {
        "tipo": "pedido_voto",
        "term": current_term,
        "candidato": NODE_ID,
        "last_version": local_version
    }
    pubsub_publish(msg)

    next_election = new_timeout()


def election_loop():
    """
    Detecta falha do líder (falta de heartbeats) e dispara eleições.
    """
    global leader_dead, next_election, state, last_heartbeat

    while True:
        time.sleep(0.2)

        # Se sou líder, não disparo eleições

        if state == "leader":
            last_heartbeat = time.time()
            continue

        # Se recebemos heartbeat recentemente, líder está vivo
        if time.time() - last_heartbeat < 5:
            continue

        if not leader_dead:
            print(f"[{NODE_ID}] LÍDER FALHOU! Preparar eleição...")
            leader_dead = True
            next_election = new_timeout()
            # Se estou desatualizado, tentar sync antes da eleição
            if need_sync():
                request_sync(force=True)

        if time.time() >= next_election:
            iniciar_eleicao()


TASK_CLAIM_DELAY_MAX = 1.5  # segundos

tasks_lock = threading.Lock()
task_owners = {}     # task_id -> peer_id
task_prompts = {}    # task_id -> prompt
task_outputs = {}    # task_id -> output (lista FAISS ou msg)


def gerar_embedding_prompt(prompt: str, task_id: str):
    """
    Reutiliza a função gerar_embedding baseada em ficheiro,
    criando temporariamente um ficheiro com o texto da prompt.
    """
    temp_path = f"temp_prompt_{task_id}.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    try:
        emb = gerar_embedding(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return emb


def process_task_search(task_id: str, prompt: str):
    """
    Este peer executa a pesquisa FAISS para a task_id.
    Output simples: top-k documentos com distância L2.
    """
    global task_outputs

    print(f"[{NODE_ID}] RF2: a processar tarefa {task_id}...")

    try:
        query_emb = gerar_embedding_prompt(prompt, task_id)
    except Exception as e:
        output = {"erro": f"Falha a gerar embedding para prompt: {str(e)}"}
        with tasks_lock:
            task_outputs[task_id] = output
        print(f"[{NODE_ID}] RF2 ERRO embedding prompt: {e}")
        return

    vetor = load_vector()
    docs = vetor.get("versoes", [])

    if not docs:
        output = {"mensagem": "Não existem documentos indexados."}
        with tasks_lock:
            task_outputs[task_id] = output
        print(f"[{NODE_ID}] RF2: sem documentos para pesquisar.")
        return

    # matriz de embeddings dos documentos
    try:
        mat = np.array([d["embedding"] for d in docs], dtype="float32")
    except Exception as e:
        output = {"erro": f"Embedding inválido nos documentos: {str(e)}"}
        with tasks_lock:
            task_outputs[task_id] = output
        print(f"[{NODE_ID}] RF2 ERRO matriz embeddings: {e}")
        return

    dim = mat.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(mat)

    q = np.array(query_emb, dtype="float32").reshape(1, -1)
    k = min(5, len(docs))
    D, I = index.search(q, k)

    resultados = []
    for rank, pos in enumerate(I[0]):
        if pos < 0 or pos >= len(docs):
            continue
        doc = docs[pos]
        resultados.append({
            "rank": rank + 1,
            "versao": doc.get("versao"),
            "cid": doc.get("cid"),
            "distance": float(D[0][rank])
        })

    output = {
        "task_id": task_id,
        "peer_id": NODE_ID,
        "resultados": resultados
    }

    with tasks_lock:
        task_outputs[task_id] = output

    print(f"[{NODE_ID}] RF2: tarefa {task_id} concluída ({len(resultados)} docs).")


def attempt_claim_and_process(task_id: str, prompt: str):
    """
    Espera um tempo aleatório e tenta reclamar a tarefa de forma distribuída.
    """
    time.sleep(random.uniform(0, TASK_CLAIM_DELAY_MAX))

    with tasks_lock:
        if task_owners.get(task_id) is not None:
            # outro peer já reclamou
            return
        task_owners[task_id] = NODE_ID

    # anunciar que este peer ficou com a tarefa
    msg = {
        "tipo": "task_claim",
        "task_id": task_id,
        "peer_id": NODE_ID
    }
    pubsub_publish(msg)
    print(f"[{NODE_ID}] RF2: reclamei a tarefa {task_id}")

    # processar FAISS
    process_task_search(task_id, prompt)


def handle_task_request(payload: dict):
    """
    RF2:
    - Recebe pedido de tarefa do líder
    - Usa backoff aleatório para decidir quem fica com a tarefa
    """
    task_id = payload.get("task_id")
    prompt = payload.get("prompt")
    if not task_id or not prompt:
        return

    with tasks_lock:
        if task_id in task_owners:
            return
        task_prompts[task_id] = prompt

    # thread separada para esperar random e tentar reclamar
    threading.Thread(
        target=attempt_claim_and_process,
        args=(task_id, prompt),
        daemon=True
    ).start()


def handle_task_claim(payload: dict):
    """
    RF2:
    - Recebemos info de quem ficou com a tarefa
    """
    task_id = payload.get("task_id")
    peer_id = payload.get("peer_id")
    if not task_id or not peer_id:
        return

    with tasks_lock:
        task_owners[task_id] = peer_id


def handle_task_output_request(payload: dict):
    """
    RF2:
    - Líder pede o resultado para task_id
    - Só o peer que tem a tarefa responde
    """
    task_id = payload.get("task_id")
    dest_peer = payload.get("peer_id")

    if not task_id:
        return

    # só responde se for o peer certo (ou se não houver dest_peer)
    if dest_peer and dest_peer != NODE_ID:
        return

    with tasks_lock:
        output = task_outputs.get(task_id)
        ready = output is not None

    msg = {
        "tipo": "task_output_response",
        "task_id": task_id,
        "peer_id": NODE_ID,
        "ready": ready
    }
    if ready:
        msg["output"] = output

    pubsub_publish(msg)
    print(f"[{NODE_ID}] RF2: output_response enviado (ready={ready}) para {task_id}")

# ------------------ LISTENER PUBSUB ------------------ #


def process_message(payload: dict):
    tipo = payload.get("tipo")

    if tipo == "heartbeat":
        receber_heartbeat(payload)
    elif tipo == "pedido_voto":
        receber_pedido_voto(payload)
    elif tipo == "voto":
        receber_voto(payload)
    elif tipo == "update_request":
        process_update_request(payload)
    elif tipo == "commit":
        process_commit(payload)
    elif tipo == "ack":
        if state == "leader":
            process_update_ack(payload)
    elif tipo == "sync_request":
        handle_sync_request(payload)
    elif tipo == "sync_response":
        handle_sync_response(payload)
    # RF2
    elif tipo == "task_request":
        handle_task_request(payload)
    elif tipo == "task_claim":
        handle_task_claim(payload)
    elif tipo == "task_output_request":
        handle_task_output_request(payload)


def listen_pubsub():
    print(f"[{NODE_ID}] A escutar tópico {TOPIC_UPDATES}...")
    proc = subprocess.Popen(
        ["ipfs", "pubsub", "sub", TOPIC_UPDATES],
        stdout=subprocess.PIPE,
        text=True
    )
    msg = ""
    nivel = 0

    while True:
        c = proc.stdout.read(1)
        if not c:
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
                    process_message(payload)
                except Exception as e:
                    print(f"[{NODE_ID}] ERRO JSON: {e}")
                msg = ""

# ------------------ FOLLOWER: UPDATE/COMMIT ------------------ #

def process_update_request(payload: dict):
    """
    FOLLOWER:
      - guarda a versão recebida no TEMP_FILE
      - envia ACK com versao + hash + NODE_ID
    """
    versao = payload["versao"]
    embedding = payload["embedding"]
    cid = payload["cid"]
    h = payload["hash"]

    vetor_temp = {
        "versoes": [{
            "versao": versao,
            "embedding": embedding,
            "cid": cid
        }]
    }
    save_temp(vetor_temp)

    msg = {
        "tipo": "ack",
        "versao": versao,
        "hash": h,
        "peer_id": NODE_ID
    }
    pubsub_publish(msg)


def process_commit(payload: dict):
    """
    FOLLOWER:
      - aplica commit usando TEMP_FILE
    """
    global cluster_max_version

    versao = payload["versao"]
    temp = load_temp()
    if temp is None:
        return

    vetor = load_vector()
    vetor["versoes"].append(temp["versoes"][0])
    save_vector(vetor)

    if versao > cluster_max_version:
        cluster_max_version = versao

    print(f"[{NODE_ID}] Commit aplicado versão {versao}")

# ------------------ LÍDER: ACKS + COMMIT ------------------ #

def commit_version(versao: int, info: dict):
    """
    LÍDER:
      - move vetor pendente para vetor_documentos.json
      - guarda também em PENDING_FILE
    """
    global cluster_max_version

    vetor_pendente = info["vector"]
    h = info["hash"]

    vetor = load_vector()
    vetor["versoes"].append(vetor_pendente["versoes"][0])
    save_vector(vetor)
    save_pending(vetor_pendente)

    if versao > cluster_max_version:
        cluster_max_version = versao

    print(f"[{NODE_ID}] (LÍDER) Versão {versao} confirmada.")

    pubsub_publish({
        "tipo": "commit",
        "versao": versao,
        "hash": h
    })


def process_update_ack(payload: dict):
    """
    LÍDER:
      - recebe ACKs de outros peers
      - confirma majority com base em CLUSTER_SIZE
    """
    versao = payload.get("versao")
    h = payload.get("hash")
    peer_id = payload.get("peer_id")

    if versao not in pending_versions:
        return

    info = pending_versions[versao]
    if h != info["hash"]:
        return

    info["acks"].add(peer_id)
    num_acks = len(info["acks"])

    # Majority baseada no nº de peers RAFT (CLUSTER_SIZE)
    majority = (CLUSTER_SIZE // 2) + 1

    print(f"[{NODE_ID}] ACK de {peer_id}: {num_acks}/{majority}")

    if num_acks >= majority:
        pending_versions.pop(versao, None)
        commit_version(versao, info)

# ------------------ LÍDER: RECUPERAÇÃO DE PENDENTES ------------------ #

def recover_pending_state():
    """
    Quando ESTE peer é eleito líder, tenta recuperar vetor_temp_lider.json
    e reenviar update_request se a versão for mais recente que o cluster.
    """
    global pending_versions, cluster_max_version

    pendente = load_pending()
    if pendente is None:
        return

    versoes = pendente.get("versoes", [])
    if not versoes:
        return

    entry = versoes[0]
    versao = entry.get("versao")
    if versao is None:
        return

    if versao <= cluster_max_version:
        return

    h = hash_vector(pendente)
    pending_versions[versao] = {"vector": pendente, "hash": h, "acks": set()}

    msg = {
        "tipo": "update_request",
        "versao": versao,
        "cid": entry["cid"],
        "embedding": entry["embedding"],
        "hash": h
    }
    pubsub_publish(msg)
    print(
        f"[{NODE_ID}] (LÍDER) Recuperação: update_request reemitido "
        f"para versão {versao}."
    )


@app.route("/add_file", methods=["POST"])
def add_file():
    """
    Endpoint para upload de ficheiros.
    Só está ativo quando ESTE peer é líder.
    """
    global pending_versions

    if state != "leader":
        return jsonify({"erro": "Este nó não é o líder atual."}), 503

    if "file" not in request.files:
        return jsonify({"erro": "Ficheiro não encontrado"}), 400

    f = request.files["file"]
    filename = f.filename or "ficheiro"
    temp_path = f"temp_{int(time.time())}_{filename}"
    f.save(temp_path)

    # 1. gerar embedding ANTES de apagar
    embedding = gerar_embedding(temp_path)

    # 2. adicionar ao IPFS
    proc = subprocess.run(
        ["ipfs", "add", "-Q", "--pin=true", temp_path],
        text=True,
        capture_output=True
    )
    os.remove(temp_path)

    if proc.returncode != 0:
        return jsonify({"erro": "Falha IPFS"}), 500

    cid = proc.stdout.strip()
    print(f"[{NODE_ID}] (LÍDER) CID = {cid}")

    vetor_atual = load_vector()
    nova_versao = len(vetor_atual["versoes"]) + 1

    nova_entry = {
        "versao": nova_versao,
        "cid": cid,
        "embedding": embedding
    }
    vetor_pendente = {"versoes": [nova_entry]}
    save_pending(vetor_pendente)
    h = hash_vector(vetor_pendente)

    pending_versions[nova_versao] = {
        "vector": vetor_pendente,
        "hash": h,
        "acks": set()
    }

    pubsub_publish({
        "tipo": "update_request",
        "versao": nova_versao,
        "cid": cid,
        "embedding": embedding,
        "hash": h
    })
    print(
        f"[{NODE_ID}] (LÍDER) update_request enviado (versão {nova_versao})."
    )

    # Commit imediato se não houver outros peers
    if CLUSTER_SIZE == 1:
        info = pending_versions.pop(nova_versao, None)
        if info:
            commit_version(nova_versao, info)

    return jsonify({
        "mensagem": "Ficheiro recebido e update enviado.",
        "cid": cid,
        "versao": nova_versao,
        "hash": h
    }), 200


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Uso: python peer_api.py <porta_http> <NODE_ID> <cluster_size>")
        sys.exit(1)

    HTTP_PORT = int(sys.argv[1])
    NODE_ID = sys.argv[2]
    CLUSTER_SIZE = int(sys.argv[3])

    cluster_max_version = get_local_last_version()

    print(f"[{NODE_ID}] Iniciado. HTTP {HTTP_PORT} cluster={CLUSTER_SIZE}")

    threading.Thread(target=listen_pubsub, daemon=True).start()
    threading.Thread(target=election_loop, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=HTTP_PORT)