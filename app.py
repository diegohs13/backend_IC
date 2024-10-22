from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertForSequenceClassification, BertConfig
from pydantic import BaseModel
import os
import torch
import pickle
import uvicorn

# Inicializar o FastAPI
app = FastAPI()

# Teste de Rota Inicial
@app.get("/")
def read_root():
    return {"mensagem": "Backend está funcionando!"}


# Configurar CORS para permitir comunicação com o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para receber os dados do formulário
class LoginData(BaseModel):
    nome_completo: str
    telefone: str

# Função para salvar dados no arquivo txt
def salvar_usuario(login_data):
    if not os.path.exists("users.txt"):
        with open("users.txt", "w") as f:
            f.write("ID | Nome Completo | Telefone\n")  # Cabeçalho

    # Gerar o próximo ID sequencial
    with open("users.txt", "r") as f:
        linhas = f.readlines()
        ultimo_id = int(linhas[-1].split(" | ")[0]) if len(linhas) > 1 else 0
        novo_id = ultimo_id + 1

    # Salvar os dados no arquivo
    with open("users.txt", "a") as f:
        f.write(f"{novo_id} | {login_data.nome_completo} | {login_data.telefone}\n")

# Rota para receber dados do login
@app.post("/login")
async def login_usuario(login_data: LoginData):
    try:
        salvar_usuario(login_data)
        return {"message": "Usuário salvo com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Configurar o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o Tokenizer
with open('modelos/tokenizer (1).pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Carregar o Modelo


# Carregar a configuração do modelo BERT
config = BertConfig.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)

# Inicializar o modelo com a configuração
model = BertForSequenceClassification(config)

# Carregar os pesos do modelo salvo em .pth
model.load_state_dict(torch.load('modelos/modelo.pth', map_location=device))

# Função para previsão
def prever_noticia(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilidade = torch.softmax(logits, dim=-1)
    predicao = torch.argmax(logits, dim=-1).item()
    return {
        "resultado": "Verdadeira" if predicao == 1 else "Falsa",
        "confiança": max(probabilidade[0]).item() * 100
    }

# Rota para previsão
@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    texto = body.get("texto", "")
    if not texto:
        return {"erro": "Nenhuma notícia foi fornecida."}
    resultado = prever_noticia(texto)
    return resultado


# Função principal para iniciar o servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Captura a variável de ambiente PORT
    uvicorn.run(app, host="0.0.0.0", port=port)