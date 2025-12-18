import sys
import os
import requests

LEADER_URL = "http://100.68.222.69:5001/add_file"


def main():
    if len(sys.argv) != 2:
        print("Uso: python client_upload.py <caminho_do_ficheiro>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Ficheiro '{file_path}' não existe.")
        sys.exit(1)

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        try:
            resp = requests.post(LEADER_URL, files=files, timeout=120)
        except Exception as e:
            print("Erro ao contactar o líder:", e)
            sys.exit(1)

    print("Código HTTP:", resp.status_code)
    try:
        print("Resposta JSON:", resp.json())
    except Exception:
        print("Resposta:", resp.text)


if __name__ == "__main__":
    main()