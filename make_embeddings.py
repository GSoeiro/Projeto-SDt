# Importa a biblioteca SentenceTransformer, usada para gerar embeddings de texto.
# Este modelo cumpre o requisito do Sprint 2: gerar embeddings para futura indexação FAISS.
from sentence_transformers import SentenceTransformer


# Carrega o modelo uma única vez (optimização).
# "all-MiniLM-L6-v2" é um modelo leve, rápido e com ótimo desempenho para embeddings.
_model = SentenceTransformer('all-MiniLM-L6-v2')


def gerar_embedding(caminho_ficheiro):
    """
    Função responsável por gerar um embedding a partir de um ficheiro.
    Tenta ler ficheiros de texto normalmente e, se falhar, tenta extrair texto de ficheiros binários.
    Retorna o embedding como uma lista de floats (compatível com JSON).
    """

    try:
        # Tenta abrir o ficheiro como texto UTF-8.
        # 'errors=ignore' evita erros com caracteres inválidos.
        with open(caminho_ficheiro, "r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

    except:
        # Caso o ficheiro não seja texto, abre como binário.
        # Lê o conteúdo e tenta converter apenas os primeiros 2000 bytes para texto.
        # Isto evita problemas com ficheiros binários grandes.
        with open(caminho_ficheiro, "rb") as f:
            data = f.read()
            texto = data[:2000].decode("latin-1", errors="ignore")

    # O modelo gera um vetor numérico (embedding) a partir do texto.
    emb = _model.encode([texto])[0]

    # Converte o vetor para lista de floats (obrigatório para JSON).
    return emb.tolist()


# Esta parte permite correr o ficheiro diretamente pelo terminal.
if __name__ == "__main__":
    import sys

    # Verifica se foi passado exatamente um argumento (o caminho do ficheiro).
    if len(sys.argv) != 2:
        print("Uso: python make_embeddings.py <ficheiro>")
        exit(1)

    caminho = sys.argv[1]

    # Gera o embedding para o ficheiro dado.
    e = gerar_embedding(caminho)

    # Mostra apenas os primeiros 10 valores — os embeddings reais têm centenas de números.
    print("Embedding gerado (10 valores iniciais):")
    print(e[:10])
