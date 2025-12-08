import datetime
import json
import uuid
import os
import math
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional

import bcrypt
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine, func, text, or_, cast, case
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# Database setup
def _get_database_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    db_file = os.getenv("DATABASE_FILE")
    if db_file:
        return f"sqlite:///{Path(db_file)}"
    return "sqlite:///./sattrack.db"


DATABASE_URL = _get_database_url()
connect_args = {"check_same_thread": False} if DATABASE_URL.lower().startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

SESSION_STORE: Dict[str, Dict[str, str]] = {}
DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@sattrack.local")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin")
try:
    LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "5"))
except Exception:
    LOW_STOCK_THRESHOLD = 5


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    codigo_interno = Column(String, unique=True, nullable=False)
    descricao = Column(String, nullable=False)
    categoria = Column(String, nullable=True)
    quantidade = Column(Integer, default=0)
    valor_unitario = Column(Float, default=0.0)
    localizacao = Column(String, nullable=True)
    foto = Column(String, nullable=True)
    observacao = Column(String, nullable=True)

    entradas = relationship("Entrada", back_populates="item")
    saidas = relationship("Saida", back_populates="item")
    transferencias = relationship("Transferencia", back_populates="item")


class Entrada(Base):
    __tablename__ = "entradas"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    quantidade = Column(Integer, nullable=False)
    usuario = Column(String, nullable=False)
    destinatario = Column(String, nullable=False, default="")
    motivo = Column(String, nullable=True)
    custo_unitario = Column(Float, nullable=True)
    nota_fiscal = Column(String, nullable=True)
    foto = Column(String, nullable=True)
    tipo = Column(String, nullable=False, default="manual")
    compra_id = Column(Integer, ForeignKey("compras.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    item = relationship("Item", back_populates="entradas")
    compra = relationship("Compra")


class Saida(Base):
    __tablename__ = "saidas"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    quantidade = Column(Integer, nullable=False)
    destino = Column(String, nullable=True)
    usuario = Column(String, nullable=False)
    motivo = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    item = relationship("Item", back_populates="saidas")


class Transferencia(Base):
    __tablename__ = "transferencias"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    origem = Column(String, nullable=True)
    destino = Column(String, nullable=False)
    quantidade = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    item = relationship("Item", back_populates="transferencias")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    operacao = Column(String, nullable=False)
    item_id = Column(Integer, nullable=True)
    quantidade = Column(Integer, nullable=True)
    usuario = Column(String, nullable=True)
    destinatario = Column(String, nullable=True)
    fornecedor = Column(String, nullable=True)
    custo = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    senha_hash = Column(String, nullable=False)
    funcao = Column(String, nullable=False, default="comum")
    data_criacao = Column(DateTime, server_default=func.now())
    foto = Column(String, nullable=True)


class Filial(Base):
    __tablename__ = "filiais"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, unique=True, nullable=False, index=True)


class Setor(Base):
    __tablename__ = "setores"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, unique=True, nullable=False, index=True)


class Fornecedor(Base):
    __tablename__ = "fornecedores"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, unique=True, nullable=False, index=True)


class Categoria(Base):
    __tablename__ = "categorias"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, unique=True, nullable=False, index=True)


class AnaliseFornecedor(Base):
    __tablename__ = "analise_fornecedores"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, unique=True, nullable=False)
    observacao = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    valores = relationship("AnaliseValor", back_populates="fornecedor", cascade="all, delete-orphan")


class AnaliseItem(Base):
    __tablename__ = "analise_itens"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    quantidade = Column(Integer, nullable=False, default=1)
    especificacao = Column(String, nullable=True)
    categoria = Column(String, nullable=True)
    tipo = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    valores = relationship("AnaliseValor", back_populates="item", cascade="all, delete-orphan")


class AnaliseValor(Base):
    __tablename__ = "analise_valores"

    id = Column(Integer, primary_key=True, index=True)
    fornecedor_id = Column(Integer, ForeignKey("analise_fornecedores.id"), nullable=False)
    item_id = Column(Integer, ForeignKey("analise_itens.id"), nullable=False)
    valor_unitario = Column(Float, nullable=False, default=0.0)
    valor_total = Column(Float, nullable=False, default=0.0)
    observacao = Column(String, nullable=True)
    documento = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    fornecedor = relationship("AnaliseFornecedor", back_populates="valores")
    item = relationship("AnaliseItem", back_populates="valores")


class CPU(Base):
    __tablename__ = "cpus"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    setor = Column(String, nullable=False)
    filial = Column(String, nullable=False)
    tag = Column(String, unique=True, nullable=False, index=True)
    processador = Column(String, nullable=False)
    memoria_ram = Column(String, nullable=False)
    placa_video = Column(String, nullable=False)
    categoria = Column(String, nullable=False)
    anydesk = Column(String, nullable=False)
    observacoes = Column(String, nullable=True)
    criado_em = Column(DateTime, server_default=func.now())
    atualizado_em = Column(DateTime, onupdate=func.now(), server_default=func.now())


class Compra(Base):
    __tablename__ = "compras"

    id = Column(Integer, primary_key=True, index=True)
    fornecedor = Column(String, nullable=False)
    categoria = Column(String, nullable=True)
    item_nome = Column(String, nullable=False)
    destino = Column(String, nullable=True)
    quantidade = Column(Integer, nullable=False, default=0)
    data_pedido = Column(String, nullable=True)
    custo_unitario = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False, default="pendente")
    caminho_nf = Column(String, nullable=True)
    data_recebimento = Column(DateTime, nullable=True)
    destinatario = Column(String, nullable=False, default="")
    usuario = Column(String, nullable=True)
    foto = Column(String, nullable=True)
    entrada_gerada = Column(Integer, nullable=False, default=0)


Base.metadata.create_all(bind=engine)


def ensure_user_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(usuarios)"))}
        alter_stmts = []
        if "email" not in cols:
            alter_stmts.append("ALTER TABLE usuarios ADD COLUMN email TEXT")
        if "senha_hash" not in cols:
            alter_stmts.append("ALTER TABLE usuarios ADD COLUMN senha_hash TEXT")
        if "funcao" not in cols:
            alter_stmts.append("ALTER TABLE usuarios ADD COLUMN funcao TEXT DEFAULT 'comum'")
        if "data_criacao" not in cols:
            alter_stmts.append("ALTER TABLE usuarios ADD COLUMN data_criacao DATETIME")
        if "foto" not in cols:
            alter_stmts.append("ALTER TABLE usuarios ADD COLUMN foto TEXT")
        for stmt in alter_stmts:
            conn.execute(text(stmt))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_usuarios_email ON usuarios(email)"))


def ensure_entrada_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(entradas)"))}
        if "custo_unitario" not in cols:
            conn.execute(text("ALTER TABLE entradas ADD COLUMN custo_unitario REAL"))
        if "nota_fiscal" not in cols:
            conn.execute(text("ALTER TABLE entradas ADD COLUMN nota_fiscal TEXT"))
        if "destinatario" not in cols:
            conn.execute(text("ALTER TABLE entradas ADD COLUMN destinatario TEXT NOT NULL DEFAULT ''"))
        if "foto" not in cols:
            conn.execute(text("ALTER TABLE entradas ADD COLUMN foto TEXT"))
        if "tipo" not in cols:
            conn.execute(text("ALTER TABLE entradas ADD COLUMN tipo TEXT NOT NULL DEFAULT 'manual'"))
        if "compra_id" not in cols:
            conn.execute(text("ALTER TABLE entradas ADD COLUMN compra_id INTEGER"))


def ensure_compras_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(compras)"))}
        if not cols:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS compras (
                        id INTEGER PRIMARY KEY,
                        fornecedor TEXT NOT NULL,
                        categoria TEXT,
                        item_nome TEXT NOT NULL,
                        destino TEXT,
                        destinatario TEXT NOT NULL DEFAULT '',
                        usuario TEXT,
                        foto TEXT,
                        quantidade INTEGER NOT NULL DEFAULT 0,
                        data_pedido TEXT,
                        custo_unitario REAL NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'pendente',
                        caminho_nf TEXT,
                        data_recebimento DATETIME,
                        entrada_gerada INTEGER NOT NULL DEFAULT 0
                    );
                    """
                )
            )
        else:
            if "destinatario" not in cols:
                conn.execute(text("ALTER TABLE compras ADD COLUMN destinatario TEXT DEFAULT ''"))
            if "usuario" not in cols:
                conn.execute(text("ALTER TABLE compras ADD COLUMN usuario TEXT"))
            if "foto" not in cols:
                conn.execute(text("ALTER TABLE compras ADD COLUMN foto TEXT"))
            if "entrada_gerada" not in cols:
                conn.execute(text("ALTER TABLE compras ADD COLUMN entrada_gerada INTEGER NOT NULL DEFAULT 0"))
            if "categoria" not in cols:
                conn.execute(text("ALTER TABLE compras ADD COLUMN categoria TEXT"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_compras_status ON compras(status)"))


def ensure_item_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(items)"))}
        if "foto" not in cols:
            conn.execute(text("ALTER TABLE items ADD COLUMN foto TEXT"))
        if "observacao" not in cols:
            conn.execute(text("ALTER TABLE items ADD COLUMN observacao TEXT"))


def ensure_audit_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(audit_log)"))}
        if "destinatario" not in cols:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN destinatario TEXT"))
        if "fornecedor" not in cols:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN fornecedor TEXT"))
        if "custo" not in cols:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN custo REAL"))


def ensure_filiais_setores_schema():
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS filiais (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT UNIQUE NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS setores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT UNIQUE NOT NULL
                );
                """
            )
        )
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_filiais_nome ON filiais(nome);"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_setores_nome ON setores(nome);"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS fornecedores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT UNIQUE NOT NULL
                );
                """
            )
        )
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_fornecedores_nome ON fornecedores(nome);"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS categorias (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT UNIQUE NOT NULL
                );
                """
            )
        )
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_categorias_nome ON categorias(nome);"))


def ensure_cpu_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(cpus)"))}
        if not cols:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS cpus (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        nome TEXT NOT NULL,
                        setor TEXT NOT NULL,
                        filial TEXT NOT NULL,
                        tag TEXT UNIQUE NOT NULL,
                        processador TEXT NOT NULL,
                        memoria_ram TEXT NOT NULL,
                        placa_video TEXT NOT NULL,
                        categoria TEXT NOT NULL,
                        anydesk TEXT NOT NULL,
                        observacoes TEXT,
                        criado_em DATETIME,
                        atualizado_em DATETIME
                    );
                    """
                )
            )
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_cpus_tag ON cpus(tag)"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def ensure_admin_user():
    db = SessionLocal()
    try:
        if not db.query(Usuario).filter(Usuario.funcao == "admin").first():
            admin = Usuario(
                nome="Administrador",
                email=DEFAULT_ADMIN_EMAIL,
                funcao="admin",
                senha_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()


ensure_item_schema()
ensure_user_schema()
ensure_admin_user()
# ensure additional schemas
ensure_entrada_schema()
ensure_compras_schema()
ensure_audit_schema()
ensure_filiais_setores_schema()
ensure_cpu_schema()

app = FastAPI(title="Sattrack Estoque")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Schemas
class ItemCreate(BaseModel):
    codigo_interno: str
    descricao: str
    categoria: Optional[str] = None
    quantidade: int = 0
    valor_unitario: float = 0.0
    localizacao: Optional[str] = None
    usuario: Optional[str] = "sistema"


class ItemUpdate(BaseModel):
    descricao: Optional[str] = None
    categoria: Optional[str] = None
    quantidade: Optional[int] = None
    valor_unitario: Optional[float] = None
    localizacao: Optional[str] = None
    usuario: Optional[str] = "sistema"


class Movimento(BaseModel):
    item_id: int
    quantidade: int
    custo_unitario: Optional[float] = None
    usuario: Optional[str] = None
    motivo: Optional[str] = None
    destinatario: Optional[str] = None
    destino: Optional[str] = None  # usado em saida


class TransferCreate(BaseModel):
    item_id: int
    origem: Optional[str] = None
    destino: str
    quantidade: int
    usuario: Optional[str] = None


class UsuarioCreate(BaseModel):
    nome: str
    funcao: str = "comum"


def get_actor(user: Dict[str, str]) -> str:
    if not user:
        return "sistema"
    return user.get("nome") or user.get("email") or "sistema"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def record_audit(
    db: Session,
    operacao: str,
    item_id: Optional[int],
    quantidade: Optional[int],
    usuario: Optional[str],
    destinatario: Optional[str] = None,
    fornecedor: Optional[str] = None,
    custo: Optional[float] = None,
):
    actor = usuario or "sistema"
    log = AuditLog(
        operacao=operacao,
        item_id=item_id,
        quantidade=quantidade,
        usuario=actor,
        destinatario=destinatario,
        fornecedor=fornecedor,
        custo=custo,
    )
    db.add(log)
    db.commit()


def record_audit_cpu(db: Session, operacao: str, usuario: str, antes: Optional[dict], depois: Optional[dict]):
    actor = usuario or "sistema"
    antes_str = json.dumps(antes or {}, ensure_ascii=False)
    depois_str = json.dumps(depois or {}, ensure_ascii=False)
    log = AuditLog(
        operacao=operacao,
        item_id=None,
        quantidade=None,
        usuario=actor,
        destinatario=antes_str[:500],
        fornecedor=depois_str[:500],
        custo=None,
    )
    db.add(log)
    db.commit()


def parse_price_to_float(raw_value: Optional[str]) -> float:
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    cleaned = str(raw_value).replace("R$", "").replace(" ", "").replace("\u00a0", "")
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        raise HTTPException(status_code=400, detail="Valor de preço inválido")


def classify_price(total: Optional[float], minimo: Optional[float], maximo: Optional[float]) -> str:
    if total is None or total <= 0:
        return "na"
    if minimo is None:
        return "mid"
    if maximo is not None and abs(maximo - minimo) < 1e-6:
        return "low"
    if abs(total - (minimo or 0)) < 1e-6:
        return "low"
    if maximo is not None and abs(total - maximo) < 1e-6:
        return "high"
    return "mid"


def build_valores_map(valores: List["AnaliseValor"]):
    mapa: Dict[tuple, AnaliseValor] = {}
    for val in valores:
        mapa[(val.item_id, val.fornecedor_id)] = val
    return mapa


def calcular_comparativo(
    itens: List["AnaliseItem"], fornecedores: List["AnaliseFornecedor"], valores_map: Dict[tuple, "AnaliseValor"]
):
    linhas = []
    total_por_fornecedor: Dict[int, float] = {f.id: 0.0 for f in fornecedores}
    fornecedor_tem_preco: Dict[int, bool] = {f.id: False for f in fornecedores}

    for item in itens:
        totais_item: List[float] = []
        for forn in fornecedores:
            val = valores_map.get((item.id, forn.id))
            if val and val.valor_total and val.valor_total > 0:
                totais_item.append(val.valor_total)
        minimo = min(totais_item) if totais_item else None
        maximo = max(totais_item) if totais_item else None

        cells = []
        for forn in fornecedores:
            val = valores_map.get((item.id, forn.id))
            total = val.valor_total if val else None
            classe = classify_price(total, minimo, maximo)
            cells.append({"fornecedor": forn, "valor": val, "classe": classe})
            if total and total > 0:
                total_por_fornecedor[forn.id] += total
                fornecedor_tem_preco[forn.id] = True

        linhas.append({"item": item, "cells": cells, "minimo": minimo, "maximo": maximo})

    ranking = [
        {
            "fornecedor": f,
            "total": total_por_fornecedor.get(f.id, 0.0),
            "tem_preco": fornecedor_tem_preco.get(f.id),
        }
        for f in fornecedores
        if fornecedor_tem_preco.get(f.id) and total_por_fornecedor.get(f.id, 0.0) > 0
    ]
    ranking = sorted(ranking, key=lambda x: x["total"])
    melhor = None
    pior = None
    if ranking:
        melhor = ranking[0]["fornecedor"]
        pior = ranking[-1]["fornecedor"]

    return linhas, total_por_fornecedor, fornecedor_tem_preco, melhor, pior


def save_uploaded_file(upload: Optional[UploadFile], folder: str, allow_pdf: bool = False) -> Optional[str]:
    if not upload or not upload.filename:
        return None
    allowed = {".jpg", ".jpeg", ".png"}
    if allow_pdf:
        allowed.add(".pdf")
    ext = Path(upload.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Formato de NF não suportado")
    target_dir = Path("app/static") / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = target_dir / filename
    file_path.write_bytes(upload.file.read())
    return f"/static/{folder}/{filename}"


def find_or_create_item_for_compra(
    db: Session,
    nome: str,
    custo_unitario: float,
    user_label: str,
    categoria: Optional[str] = None,
    foto: Optional[str] = None,
) -> Item:
    nome_norm = nome.strip()
    candidatos = db.query(Item).filter(func.lower(Item.descricao) == nome_norm.lower()).all()

    def custo_igual(it: Item) -> bool:
        if custo_unitario is None:
            return True
        if it.valor_unitario is None:
            return False
        return abs((it.valor_unitario or 0) - custo_unitario) < 1e-6

    for it in candidatos:
        if custo_igual(it):
            updated = False
            if categoria and (not it.categoria or it.categoria.lower() == "sem categoria"):
                it.categoria = categoria
                updated = True
            if foto and not it.foto:
                it.foto = foto
                updated = True
            if custo_unitario and it.valor_unitario != custo_unitario:
                it.valor_unitario = custo_unitario
                updated = True
            if updated:
                db.commit()
            return it

    base_codigo = "".join(ch for ch in nome.upper().replace(" ", "_") if ch.isalnum() or ch == "_") or "ITEM"
    codigo = base_codigo
    suffix = 1
    while db.query(Item).filter(Item.codigo_interno == codigo).first():
        suffix += 1
        codigo = f"{base_codigo}_{suffix}"
    item = Item(
        codigo_interno=codigo,
        descricao=nome,
        categoria=categoria or "Sem Categoria",
        quantidade=0,
        valor_unitario=custo_unitario or 0.0,
        localizacao="Almoxarifado",
        foto=foto,
    )
    db.add(item)
    db.commit()
    record_audit(db, "cadastro_item_por_compra", item.id, 0, user_label, custo=custo_unitario)
    return item


def process_entrega_compra(db: Session, compra: Compra, user_label: str):
    if compra.entrada_gerada:
        return
    item = find_or_create_item_for_compra(
        db,
        compra.item_nome,
        compra.custo_unitario,
        user_label,
        categoria=compra.categoria,
        foto=compra.foto,
    )
    item.quantidade += compra.quantidade
    if compra.custo_unitario:
        item.valor_unitario = compra.custo_unitario
    entrada = Entrada(
        item_id=item.id,
        quantidade=compra.quantidade,
        usuario=user_label,
        destinatario=compra.destinatario or "",
        motivo=f"Compra de {compra.fornecedor}",
        custo_unitario=compra.custo_unitario,
        nota_fiscal=compra.caminho_nf,
        foto=compra.foto,
        tipo="automatica",
        compra_id=compra.id,
    )
    db.add(entrada)
    compra.entrada_gerada = 1
    compra.data_recebimento = compra.data_recebimento or datetime.datetime.utcnow()
    db.commit()
    record_audit(
        db,
        "estoque_atualizado_compra",
        item.id,
        compra.quantidade,
        user_label,
        destinatario=compra.destinatario,
        fornecedor=compra.fornecedor,
        custo=compra.custo_unitario,
    )
    record_audit(
        db,
        "entrada_compra",
        item.id,
        compra.quantidade,
        user_label,
        destinatario=compra.destinatario,
        fornecedor=compra.fornecedor,
        custo=compra.custo_unitario,
    )
    return item


def create_session(user: Usuario) -> str:
    token = uuid.uuid4().hex
    SESSION_STORE[token] = {
        "id": user.id,
        "nome": user.nome,
        "funcao": user.funcao,
        "email": user.email,
        "foto": user.foto,
    }
    return token


def clear_session(token: Optional[str]):
    if token and token in SESSION_STORE:
        del SESSION_STORE[token]


def require_user(session: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    session_data = SESSION_STORE.get(session or "")
    if not session_data:
        raise HTTPException(status_code=303, detail="login requerido", headers={"Location": "/login"})
    db_user = db.get(Usuario, session_data["id"])
    if not db_user:
        clear_session(session)
        raise HTTPException(status_code=303, detail="login requerido", headers={"Location": "/login"})
    return {
        "id": db_user.id,
        "nome": db_user.nome,
        "funcao": db_user.funcao,
        "email": db_user.email,
        "foto": db_user.foto,
    }


def require_admin(user=Depends(require_user)):
    if user["funcao"] != "admin":
        raise HTTPException(status_code=303, detail="apenas admin", headers={"Location": "/dashboard"})
    return user


@app.get("/", include_in_schema=False)
def root(user=Depends(require_user)):
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/dashboard", include_in_schema=False)
def dashboard(request: Request, db: Session = Depends(get_db), user=Depends(require_user)):
    total_itens = db.query(Item).count()
    total_quantidade = db.query(func.coalesce(func.sum(Item.quantidade), 0)).scalar() or 0
    valor_total = sum((it.quantidade or 0) * (it.valor_unitario or 0) for it in db.query(Item).all())
    estoque_baixo = db.query(Item).filter(Item.quantidade <= LOW_STOCK_THRESHOLD).count()
    inicio_hoje = datetime.datetime.combine(datetime.datetime.utcnow().date(), datetime.time.min)
    fim_hoje = inicio_hoje + datetime.timedelta(days=1)
    movimentacoes_hoje = (
        db.query(Entrada).filter(Entrada.created_at >= inicio_hoje, Entrada.created_at < fim_hoje).count()
        + db.query(Saida).filter(Saida.created_at >= inicio_hoje, Saida.created_at < fim_hoje).count()
    )
    ultimas_entradas = db.query(Entrada).order_by(Entrada.created_at.desc()).limit(5).all()
    ultimas_saidas = db.query(Saida).order_by(Saida.created_at.desc()).limit(5).all()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "total_itens": total_itens,
            "total_quantidade": total_quantidade,
            "valor_total": valor_total,
            "estoque_baixo": estoque_baixo,
            "movimentacoes_hoje": movimentacoes_hoje,
            "ultimas_entradas": ultimas_entradas,
            "ultimas_saidas": ultimas_saidas,
            "user": user,
        },
    )


@app.get("/api/dashboard/estoque")
def dashboard_estoque(db: Session = Depends(get_db), user=Depends(require_user)):
    itens = db.query(Item).order_by(Item.descricao).all()
    return {
        "labels": [i.descricao for i in itens],
        "quantidades": [i.quantidade for i in itens],
    }


@app.get("/api/dashboard/fluxo")
def dashboard_fluxo(db: Session = Depends(get_db), user=Depends(require_user)):
    hoje = datetime.date.today()
    labels: List[str] = []
    entradas: List[int] = []
    saidas: List[int] = []
    for i in range(6, -1, -1):
        dia = hoje - datetime.timedelta(days=i)
        total_entradas = (
            db.query(func.coalesce(func.sum(Entrada.quantidade), 0))
            .filter(func.date(Entrada.created_at) == dia.isoformat())
            .scalar()
            or 0
        )
        total_saidas = (
            db.query(func.coalesce(func.sum(Saida.quantidade), 0))
            .filter(func.date(Saida.created_at) == dia.isoformat())
            .scalar()
            or 0
        )
        labels.append(dia.strftime("%d/%m"))
        entradas.append(int(total_entradas))
        saidas.append(int(total_saidas))
    return {"labels": labels, "entradas": entradas, "saidas": saidas}


@app.get("/api/dashboard/categorias")
def dashboard_categorias(db: Session = Depends(get_db), user=Depends(require_user)):
    rows = (
        db.query(Item.categoria, func.coalesce(func.sum(Item.quantidade), 0))
        .group_by(Item.categoria)
        .all()
    )
    labels: List[str] = []
    quantidades: List[int] = []
    for categoria, total in rows:
        labels.append(categoria or "Sem categoria")
        quantidades.append(int(total or 0))
    return {"labels": labels, "quantidades": quantidades}


# Inventory API
@app.get("/api/items", response_model=List[ItemCreate])
def list_items(db: Session = Depends(get_db), user=Depends(require_user)):
    return db.query(Item).all()


@app.post("/api/items", response_model=ItemCreate)
def create_item_api(payload: ItemCreate, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    if db.query(Item).filter(Item.codigo_interno == payload.codigo_interno).first():
        raise HTTPException(status_code=400, detail="Codigo interno ja existe")
    item = Item(
        codigo_interno=payload.codigo_interno,
        descricao=payload.descricao,
        categoria=payload.categoria,
        quantidade=payload.quantidade,
        valor_unitario=payload.valor_unitario,
        localizacao=payload.localizacao,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    record_audit(db, "cadastro", item.id, payload.quantidade, actor)
    return item


@app.put("/api/items/{item_id}", response_model=ItemCreate)
def update_item_api(item_id: int, payload: ItemUpdate, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    qty_before = item.quantidade
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(item, field, value)
    db.commit()
    db.refresh(item)
    delta = item.quantidade - qty_before if payload.quantidade is not None else 0
    record_audit(db, "edicao", item.id, delta, actor)
    return item


@app.delete("/api/items/{item_id}")
def delete_item_api(item_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    db.delete(item)
    db.commit()
    record_audit(db, "exclusao", item_id, None, actor)
    return {"status": "ok"}


# Controle de CPUs
@app.get("/controle-cpus", include_in_schema=False)
def controle_cpus_page(
    request: Request,
    setor: str = "",
    filial: str = "",
    categoria: str = "",
    nome: str = "",
    tag: str = "",
    busca: str = "",
    sort: str = "nome",
    dir: str = "asc",
    page: int = 1,
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    query = db.query(CPU)
    if setor:
        query = query.filter(CPU.setor.ilike(f"%{setor}%"))
    if filial:
        query = query.filter(CPU.filial.ilike(f"%{filial}%"))
    if categoria:
        query = query.filter(CPU.categoria == categoria)
    if nome:
        query = query.filter(CPU.nome.ilike(f"%{nome}%"))
    if tag:
        query = query.filter(CPU.tag.ilike(f"%{tag}%"))
    if busca:
        busca_like = f"%{busca}%"
        query = query.filter(
            CPU.nome.ilike(busca_like) | CPU.tag.ilike(busca_like) | CPU.processador.ilike(busca_like)
        )
    sort = (sort or "nome").lower()
    dir = (dir or "asc").lower()
    if sort not in {"nome", "tag", "categoria", "setor"}:
        sort = "nome"
    if dir not in {"asc", "desc"}:
        dir = "asc"
    if sort == "nome":
        sort_column = func.lower(CPU.nome)
        tie_breakers = (CPU.tag, CPU.categoria, CPU.nome)
    elif sort == "tag":
        sort_column = func.lower(CPU.tag)
        tie_breakers = (CPU.nome, CPU.categoria, CPU.tag)
    elif sort == "categoria":
        categoria_order = case(
            (CPU.categoria == "NOVA-FORTE", 1),
            (CPU.categoria == "NOVA", 2),
            (CPU.categoria == "FRACA-ANTIGA", 3),
            else_=4,
        )
        # usa a ordem customizada e baixa para desempate consistente
        sort_column = categoria_order
        tie_breakers = (func.lower(CPU.categoria), CPU.nome, CPU.tag)
    else:  # setor
        sort_column = func.lower(CPU.setor)
        tie_breakers = (CPU.nome, CPU.tag, CPU.categoria)
    if dir == "asc":
        query = query.order_by(
            sort_column.asc(),
            *[col.asc() for col in tie_breakers],
        )
    else:
        query = query.order_by(
            sort_column.desc(),
            *[col.desc() for col in tie_breakers],
        )

    page = max(page, 1)
    per_page = 15
    total = query.count()
    cpus = query.offset((page - 1) * per_page).limit(per_page).all()

    setores = [row[0] for row in db.query(CPU.setor).filter(CPU.setor.isnot(None)).distinct().order_by(CPU.setor)]
    filiais = [row[0] for row in db.query(CPU.filial).filter(CPU.filial.isnot(None)).distinct().order_by(CPU.filial)]
    categorias = ["NOVA-FORTE", "NOVA", "FRACA-ANTIGA"]
    filiais_options = [f.nome for f in db.query(Filial).order_by(Filial.nome).all()]
    setores_options = [s.nome for s in db.query(Setor).order_by(Setor.nome).all()]

    categoria_counts = {row[0]: row[1] for row in db.query(CPU.categoria, func.count()).group_by(CPU.categoria)}
    filial_counts = {row[0]: row[1] for row in db.query(CPU.filial, func.count()).group_by(CPU.filial)}

    return templates.TemplateResponse(
        "controle_cpus.html",
        {
            "request": request,
            "user": user,
            "cpus": cpus,
            "setores": setores,
            "filiais": filiais,
            "categorias": categorias,
            "filtro_setor": setor,
            "filtro_filial": filial,
            "filtro_categoria": categoria,
            "filtro_nome": nome,
            "filtro_tag": tag,
            "filtro_busca": busca,
            "filtro_sort": sort,
            "filtro_dir": dir,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page if per_page else 1,
            "erro": request.query_params.get("erro"),
            "categoria_counts": categoria_counts,
            "filial_counts": filial_counts,
            "filiais_options": filiais_options,
            "setores_options": setores_options,
        },
    )


@app.post("/controle-cpus/criar", include_in_schema=False)
def controle_cpus_criar(
    nome: str = Form(...),
    setor: str = Form(...),
    filial: str = Form(...),
    tag: str = Form(...),
    processador: str = Form(...),
    memoria_ram: str = Form(...),
    placa_video: str = Form(...),
    categoria: str = Form(...),
    anydesk: str = Form(...),
    observacoes: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    if db.query(CPU).filter(CPU.tag == tag).first():
        return RedirectResponse(url="/controle-cpus?erro=tag", status_code=303)
    novo = CPU(
        nome=nome,
        setor=setor,
        filial=filial,
        tag=tag,
        processador=processador,
        memoria_ram=memoria_ram,
        placa_video=placa_video,
        categoria=categoria,
        anydesk=anydesk,
        observacoes=observacoes,
    )
    db.add(novo)
    db.commit()
    record_audit_cpu(db, "criar_cpu", actor, None, {
        "nome": nome,
        "setor": setor,
        "filial": filial,
        "tag": tag,
        "processador": processador,
        "memoria_ram": memoria_ram,
        "placa_video": placa_video,
        "categoria": categoria,
        "anydesk": anydesk,
        "observacoes": observacoes,
    })
    return RedirectResponse(url="/controle-cpus", status_code=303)


@app.post("/controle-cpus/{cpu_id}/editar", include_in_schema=False)
def controle_cpus_editar(
    cpu_id: int,
    nome: str = Form(...),
    setor: str = Form(...),
    filial: str = Form(...),
    tag: str = Form(...),
    processador: str = Form(...),
    memoria_ram: str = Form(...),
    placa_video: str = Form(...),
    categoria: str = Form(...),
    anydesk: str = Form(...),
    observacoes: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    cpu = db.get(CPU, cpu_id)
    if not cpu:
        raise HTTPException(status_code=404, detail="CPU não encontrada")
    if tag != cpu.tag and db.query(CPU).filter(CPU.tag == tag).first():
        return RedirectResponse(url="/controle-cpus?erro=tag", status_code=303)
    antes = {
        "nome": cpu.nome,
        "setor": cpu.setor,
        "filial": cpu.filial,
        "tag": cpu.tag,
        "processador": cpu.processador,
        "memoria_ram": cpu.memoria_ram,
        "placa_video": cpu.placa_video,
        "categoria": cpu.categoria,
        "anydesk": cpu.anydesk,
        "observacoes": cpu.observacoes,
    }
    cpu.nome = nome
    cpu.setor = setor
    cpu.filial = filial
    cpu.tag = tag
    cpu.processador = processador
    cpu.memoria_ram = memoria_ram
    cpu.placa_video = placa_video
    cpu.categoria = categoria
    cpu.anydesk = anydesk
    cpu.observacoes = observacoes
    db.commit()
    depois = {
        "nome": cpu.nome,
        "setor": cpu.setor,
        "filial": cpu.filial,
        "tag": cpu.tag,
        "processador": cpu.processador,
        "memoria_ram": cpu.memoria_ram,
        "placa_video": cpu.placa_video,
        "categoria": cpu.categoria,
        "anydesk": cpu.anydesk,
        "observacoes": cpu.observacoes,
    }
    record_audit_cpu(db, "editar_cpu", actor, antes, depois)
    return RedirectResponse(url="/controle-cpus", status_code=303)


@app.post("/controle-cpus/{cpu_id}/deletar", include_in_schema=False)
def controle_cpus_deletar(cpu_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    cpu = db.get(CPU, cpu_id)
    if not cpu:
        raise HTTPException(status_code=404, detail="CPU não encontrada")
    antes = {
        "nome": cpu.nome,
        "setor": cpu.setor,
        "filial": cpu.filial,
        "tag": cpu.tag,
        "processador": cpu.processador,
        "memoria_ram": cpu.memoria_ram,
        "placa_video": cpu.placa_video,
        "categoria": cpu.categoria,
        "anydesk": cpu.anydesk,
        "observacoes": cpu.observacoes,
    }
    db.delete(cpu)
    db.commit()
    record_audit_cpu(db, "deletar_cpu", actor, antes, None)
    return RedirectResponse(url="/controle-cpus", status_code=303)

# Filiais e Setores (cadastro simples)
@app.get("/filiais-setores", include_in_schema=False)
def filiais_setores_page(request: Request, db: Session = Depends(get_db), user=Depends(require_admin)):
    filiais = db.query(Filial).order_by(Filial.nome).all()
    setores = db.query(Setor).order_by(Setor.nome).all()
    fornecedores = db.query(Fornecedor).order_by(Fornecedor.nome).all()
    categorias = db.query(Categoria).order_by(Categoria.nome).all()
    return templates.TemplateResponse(
        "filiais_setores.html",
        {
            "request": request,
            "user": user,
            "filiais": filiais,
            "setores": setores,
            "fornecedores": fornecedores,
            "categorias": categorias,
            "erro": request.query_params.get("erro"),
        },
    )


@app.post("/filiais/criar", include_in_schema=False)
def criar_filial(nome: str = Form(...), db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    if db.query(Filial).filter(func.lower(Filial.nome) == nome.lower()).first():
        return RedirectResponse(url="/filiais-setores?erro=filial", status_code=303)
    nova = Filial(nome=nome)
    db.add(nova)
    db.commit()
    record_audit(db, "criar_filial", None, None, actor, fornecedor=nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)


@app.post("/setores/criar", include_in_schema=False)
def criar_setor(nome: str = Form(...), db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    if db.query(Setor).filter(func.lower(Setor.nome) == nome.lower()).first():
        return RedirectResponse(url="/filiais-setores?erro=setor", status_code=303)
    novo = Setor(nome=nome)
    db.add(novo)
    db.commit()
    record_audit(db, "criar_setor", None, None, actor, fornecedor=nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)


@app.post("/filiais/{filial_id}/deletar", include_in_schema=False)
def deletar_filial(filial_id: int, db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    filial = db.get(Filial, filial_id)
    if not filial:
        raise HTTPException(status_code=404, detail="Filial não encontrada")
    db.delete(filial)
    db.commit()
    record_audit(db, "deletar_filial", None, None, actor, fornecedor=filial.nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)


@app.post("/setores/{setor_id}/deletar", include_in_schema=False)
def deletar_setor(setor_id: int, db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    setor = db.get(Setor, setor_id)
    if not setor:
        raise HTTPException(status_code=404, detail="Setor não encontrado")
    db.delete(setor)
    db.commit()
    record_audit(db, "deletar_setor", None, None, actor, fornecedor=setor.nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)


@app.post("/fornecedores/criar", include_in_schema=False)
def criar_fornecedor(nome: str = Form(...), db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    if db.query(Fornecedor).filter(func.lower(Fornecedor.nome) == nome.lower()).first():
        return RedirectResponse(url="/filiais-setores?erro=fornecedor", status_code=303)
    novo = Fornecedor(nome=nome)
    db.add(novo)
    db.commit()
    record_audit(db, "criar_fornecedor", None, None, actor, fornecedor=nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)


@app.post("/fornecedores/{fornecedor_id}/deletar", include_in_schema=False)
def deletar_fornecedor(fornecedor_id: int, db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    fornecedor = db.get(Fornecedor, fornecedor_id)
    if not fornecedor:
        raise HTTPException(status_code=404, detail="Fornecedor não encontrado")
    db.delete(fornecedor)
    db.commit()
    record_audit(db, "deletar_fornecedor", None, None, actor, fornecedor=fornecedor.nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)

# Análise de Valores de Produtos
@app.post("/categorias/criar", include_in_schema=False)
def criar_categoria(nome: str = Form(...), db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    if db.query(Categoria).filter(func.lower(Categoria.nome) == nome.lower()).first():
        return RedirectResponse(url="/filiais-setores?erro=categoria", status_code=303)
    nova = Categoria(nome=nome)
    db.add(nova)
    db.commit()
    record_audit(db, "criar_categoria", None, None, actor, fornecedor=nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)

@app.post("/categorias/{categoria_id}/deletar", include_in_schema=False)
def deletar_categoria(categoria_id: int, db: Session = Depends(get_db), user=Depends(require_admin)):
    actor = get_actor(user)
    cat = db.get(Categoria, categoria_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Categoria nǜo encontrada")
    db.delete(cat)
    db.commit()
    record_audit(db, "deletar_categoria", None, None, actor, fornecedor=cat.nome)
    return RedirectResponse(url="/filiais-setores", status_code=303)

@app.get("/analise-valores", include_in_schema=False)
def analise_valores_page(
    request: Request,
    categoria: str = "",
    tipo: str = "",
    fornecedor: str = "",
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    itens_query = db.query(AnaliseItem)
    if categoria:
        itens_query = itens_query.filter(AnaliseItem.categoria.ilike(f"%{categoria}%"))
    if tipo:
        itens_query = itens_query.filter(AnaliseItem.tipo.ilike(f"%{tipo}%"))
    itens = itens_query.order_by(AnaliseItem.nome).all()

    fornecedores_base = db.query(Fornecedor).order_by(Fornecedor.nome).all()
    # garante espelho em analise_fornecedores a partir de fornecedores base
    fornecedores_map: Dict[int, AnaliseFornecedor] = {}
    for f in fornecedores_base:
        existente = db.query(AnaliseFornecedor).filter(func.lower(AnaliseFornecedor.nome) == f.nome.lower()).first()
        if not existente:
            existente = AnaliseFornecedor(nome=f.nome)
            db.add(existente)
            db.commit()
        fornecedores_map[f.id] = existente
    fornecedores_view = [fornecedores_map[f.id] for f in fornecedores_base if (not fornecedor or fornecedor.lower() in f.nome.lower())]

    valores = db.query(AnaliseValor).all()
    valores_map = build_valores_map(valores)

    linhas, total_por_fornecedor, fornecedor_tem_preco, melhor, pior = calcular_comparativo(
        itens, fornecedores_view, valores_map
    )
    ranking = [
        {
            "fornecedor": f,
            "total": total_por_fornecedor.get(f.id, 0.0),
            "tem_preco": fornecedor_tem_preco.get(f.id),
        }
        for f in fornecedores_view
        if fornecedor_tem_preco.get(f.id) and total_por_fornecedor.get(f.id, 0.0) > 0
    ]
    ranking = sorted(ranking, key=lambda x: x["total"])
    melhor = ranking[0]["fornecedor"] if ranking else None
    pior = ranking[-1]["fornecedor"] if ranking else None
    totais_validos = [total_por_fornecedor.get(f.id, 0.0) for f in fornecedores_view if fornecedor_tem_preco.get(f.id) and total_por_fornecedor.get(f.id, 0.0) > 0]
    min_total = min(totais_validos) if totais_validos else None
    max_total = max(totais_validos) if totais_validos else None
    total_classes = {f.id: classify_price(total_por_fornecedor.get(f.id), min_total, max_total) for f in fornecedores_view}
    quantidade_total = sum(i.quantidade or 0 for i in itens)
    categorias_options = [
        row[0] for row in db.query(AnaliseItem.categoria).filter(AnaliseItem.categoria.isnot(None)).distinct()
    ]
    tipos_options = [row[0] for row in db.query(AnaliseItem.tipo).filter(AnaliseItem.tipo.isnot(None)).distinct()]

    return templates.TemplateResponse(
        "analise_valores.html",
        {
            "request": request,
            "user": user,
            "itens": itens,
            "fornecedores": fornecedores_view,
            "fornecedores_base": fornecedores_base,
            "valores_map": valores_map,
            "linhas": linhas,
            "total_por_fornecedor": total_por_fornecedor,
            "fornecedor_tem_preco": fornecedor_tem_preco,
            "melhor_fornecedor_total": melhor,
            "pior_fornecedor_total": pior,
            "ranking": ranking,
            "min_total": min_total,
            "max_total": max_total,
            "total_classes": total_classes,
            "quantidade_total": quantidade_total,
            "categorias_options": categorias_options,
            "tipos_options": tipos_options,
            "filtro_categoria": categoria,
            "filtro_tipo": tipo,
            "filtro_fornecedor": fornecedor,
        },
    )


@app.post("/analise-valores/itens/criar", include_in_schema=False)
def analise_item_criar(
    nome: str = Form(...),
    quantidade: int = Form(...),
    especificacao: str = Form(""),
    categoria: str = Form(""),
    tipo: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    novo = AnaliseItem(
        nome=nome.strip(),
        quantidade=max(1, quantidade),
        especificacao=especificacao,
        categoria=categoria or None,
        tipo=tipo or None,
    )
    db.add(novo)
    db.commit()
    record_audit(db, "analise_criar_item", novo.id, quantidade, actor)
    return RedirectResponse(url="/analise-valores", status_code=303)


@app.post("/analise-valores/itens/{item_id}/deletar", include_in_schema=False)
def analise_item_deletar(item_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(AnaliseItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item não encontrado")
    nome_item = item.nome
    db.delete(item)
    db.commit()
    record_audit(db, "analise_deletar_item", item_id, None, actor, fornecedor=nome_item)
    return RedirectResponse(url="/analise-valores", status_code=303)


@app.post("/analise-valores/valores/criar", include_in_schema=False)
def analise_valor_criar(
    fornecedor_id: int = Form(...),
    item_id: int = Form(...),
    valor_unitario: str = Form(...),
    observacao: str = Form(""),
    documento: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    fornecedor_base = db.get(Fornecedor, fornecedor_id)
    item = db.get(AnaliseItem, item_id)
    if not fornecedor_base or not item:
        raise HTTPException(status_code=404, detail="Fornecedor ou item não encontrado")
    fornecedor_obj = db.query(AnaliseFornecedor).filter(func.lower(AnaliseFornecedor.nome) == fornecedor_base.nome.lower()).first()
    if not fornecedor_obj:
        fornecedor_obj = AnaliseFornecedor(nome=fornecedor_base.nome)
        db.add(fornecedor_obj)
        db.commit()
    valor_unitario_float = parse_price_to_float(valor_unitario)
    valor_total = round(valor_unitario_float * (item.quantidade or 0), 2)
    doc_path = save_uploaded_file(documento, f"uploads/analise_valores/{item_id}", allow_pdf=True)
    novo = AnaliseValor(
        fornecedor_id=fornecedor_obj.id,
        item_id=item.id,
        valor_unitario=valor_unitario_float,
        valor_total=valor_total,
        observacao=observacao,
        documento=doc_path,
    )
    db.add(novo)
    db.commit()
    record_audit(
        db,
        "analise_valor_criar",
        item.id,
        item.quantidade,
        actor,
        fornecedor=fornecedor_obj.nome,
        custo=valor_unitario_float,
    )
    return RedirectResponse(url="/analise-valores", status_code=303)


@app.post("/analise-valores/valores/{valor_id}/deletar", include_in_schema=False)
def analise_valor_deletar(valor_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    valor = db.get(AnaliseValor, valor_id)
    if not valor:
        raise HTTPException(status_code=404, detail="Valor não encontrado")
    fornecedor_nome = valor.fornecedor.nome if valor.fornecedor else None
    item_rel = valor.item_id
    db.delete(valor)
    db.commit()
    record_audit(db, "analise_valor_deletar", item_rel, None, actor, fornecedor=fornecedor_nome)
    return RedirectResponse(url="/analise-valores", status_code=303)

# Inventory pages
@app.get("/inventario", include_in_schema=False)
def inventory_page(
    request: Request,
    codigo: str = "",
    descricao: str = "",
    categoria: str = "",
    page: int = 1,
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    query = db.query(Item)
    if codigo:
        query = query.filter(Item.codigo_interno.ilike(f"%{codigo}%"))
    if descricao:
        query = query.filter(Item.descricao.ilike(f"%{descricao}%"))
    if categoria:
        query = query.filter(Item.categoria.ilike(f"%{categoria}%"))
    page = max(page, 1)
    per_page = 10
    total = query.count()
    total_pages = max((total + per_page - 1) // per_page, 1)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    itens = (
        query.order_by(Item.descricao)
        .offset(start)
        .limit(per_page)
        .all()
    )
    erro_item_id = request.query_params.get("item_id")
    item_relacionado = None
    if erro_item_id:
        try:
            item_relacionado = db.get(Item, int(erro_item_id))
        except Exception:
            item_relacionado = None
    categorias = [c.nome for c in db.query(Categoria).order_by(Categoria.nome).all()]
    return templates.TemplateResponse(
        "inventario.html",
        {
            "request": request,
            "itens": itens,
            "user": user,
            "filtro_codigo": codigo,
            "filtro_descricao": descricao,
            "filtro_categoria": categoria,
            "categorias": categorias,
            "erro": request.query_params.get("erro"),
            "item_relacionado": item_relacionado,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "start": start,
        },
    )


@app.post("/inventario/cadastrar", include_in_schema=False)
async def cadastrar_item_inventario(
    codigo_interno: str = Form(...),
    descricao: str = Form(...),
    categoria: str = Form(""),
    quantidade: int = Form(0),
    valor_unitario: float = Form(0.0),
    localizacao: str = Form(""),
    foto: UploadFile = File(None),
    observacao: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    if db.query(Item).filter(Item.codigo_interno == codigo_interno).first():
        return RedirectResponse(url="/inventario?erro=codigo", status_code=303)
    foto_path = save_uploaded_file(foto, "uploads/itens")
    item = Item(
        codigo_interno=codigo_interno,
        descricao=descricao,
        categoria=categoria,
        quantidade=quantidade,
        valor_unitario=valor_unitario,
        localizacao=localizacao,
        foto=foto_path,
        observacao=observacao,
    )
    db.add(item)
    db.commit()
    record_audit(db, "cadastro", item.id, quantidade, actor)
    return RedirectResponse(url=f"/movimentacoes/entradas?item_id={item.id}", status_code=303)


@app.post("/items/criar", include_in_schema=False)
async def create_item_form(
    codigo_interno: str = Form(...),
    descricao: str = Form(...),
    categoria: str = Form(""),
    valor_unitario: float = Form(0.0),
    localizacao: str = Form(""),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    if db.query(Item).filter(Item.codigo_interno == codigo_interno).first():
        raise HTTPException(status_code=400, detail="Codigo interno ja existe")
    foto_path = save_uploaded_file(foto, "uploads/itens")
    item = Item(
        codigo_interno=codigo_interno,
        descricao=descricao,
        categoria=categoria,
        quantidade=0,
        valor_unitario=valor_unitario,
        localizacao=localizacao,
        foto=foto_path,
    )
    db.add(item)
    db.commit()
    record_audit(db, "cadastro", item.id, 0, actor)
    return RedirectResponse(url="/inventario", status_code=303)


@app.post("/items/{item_id}/editar", include_in_schema=False)
async def edit_item_form(
    item_id: int,
    descricao: str = Form(...),
    categoria: str = Form(""),
    quantidade: int = Form(...),
    valor_unitario: float = Form(0.0),
    localizacao: str = Form(""),
    foto: UploadFile = File(None),
    observacao: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    delta = quantidade - item.quantidade
    item.descricao = descricao
    item.categoria = categoria
    item.quantidade = quantidade
    item.valor_unitario = valor_unitario
    item.localizacao = localizacao
    item.observacao = observacao
    foto_path = save_uploaded_file(foto, "uploads/itens")
    if foto_path:
        item.foto = foto_path
    db.commit()
    record_audit(db, "edicao", item.id, delta, actor)
    return RedirectResponse(url="/inventario", status_code=303)


@app.post("/items/{item_id}/deletar", include_in_schema=False)
def delete_item_form(
    item_id: int,
    force: int = Form(0),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    # Impede exclusão se o item tiver movimentações vinculadas, a menos que forçado
    tem_entradas = db.query(Entrada.id).filter(Entrada.item_id == item_id).first()
    tem_saidas = db.query(Saida.id).filter(Saida.item_id == item_id).first()
    tem_transf = db.query(Transferencia.id).filter(Transferencia.item_id == item_id).first()
    if (tem_entradas or tem_saidas or tem_transf) and not force:
        return RedirectResponse(url=f"/inventario?erro=item_relacionado&item_id={item_id}", status_code=303)
    if force:
        db.query(Entrada).filter(Entrada.item_id == item_id).delete(synchronize_session=False)
        db.query(Saida).filter(Saida.item_id == item_id).delete(synchronize_session=False)
        db.query(Transferencia).filter(Transferencia.item_id == item_id).delete(synchronize_session=False)
    db.delete(item)
    db.commit()
    record_audit(db, "exclusao_forcada" if force else "exclusao", item_id, None, actor)
    return RedirectResponse(url="/inventario", status_code=303)


# Entradas
@app.get("/entradas", include_in_schema=False)
def entradas_page(request: Request, item_id: Optional[int] = None, db: Session = Depends(get_db), user=Depends(require_user)):
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


@app.post("/api/entradas")
def registrar_entrada_api(payload: Movimento, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(Item, payload.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    if not payload.destinatario:
        raise HTTPException(status_code=400, detail="Destinatario obrigatorio")
    item.quantidade += payload.quantidade
    if payload.custo_unitario:
        item.valor_unitario = payload.custo_unitario
    entrada = Entrada(
        item_id=item.id,
        quantidade=payload.quantidade,
        usuario=actor,
        destinatario=payload.destinatario,
        motivo=payload.motivo,
        custo_unitario=payload.custo_unitario,
        tipo="manual",
    )
    db.add(entrada)
    db.commit()
    record_audit(
        db,
        "entrada",
        item.id,
        payload.quantidade,
        actor,
        destinatario=payload.destinatario,
        custo=payload.custo_unitario,
    )
    return {"status": "ok"}

@app.post("/entradas/registrar", include_in_schema=False)
async def registrar_entrada_form(
    item_id: int = Form(...),
    quantidade: int = Form(...),
    destinatario: str = Form(...),
    motivo: str = Form(""),
    custo_unitario: float = Form(0.0),
    nota_fiscal: UploadFile = File(None),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    if not destinatario:
        raise HTTPException(status_code=400, detail="Destinatario obrigatorio")
    item.quantidade += quantidade
    if custo_unitario:
        item.valor_unitario = custo_unitario
    nf_path = save_uploaded_file(nota_fiscal, "nf/entradas", allow_pdf=True)
    foto_path = save_uploaded_file(foto, "uploads/entradas")
    entrada = Entrada(
        item_id=item.id,
        quantidade=quantidade,
        usuario=actor,
        destinatario=destinatario,
        motivo=motivo,
        custo_unitario=custo_unitario or item.valor_unitario,
        nota_fiscal=nf_path,
        foto=foto_path,
        tipo="manual",
    )
    db.add(entrada)
    db.commit()
    record_audit(
        db,
        "entrada",
        item.id,
        quantidade,
        actor,
        destinatario=destinatario,
        custo=custo_unitario,
    )
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


@app.post("/entradas/{entrada_id}/editar", include_in_schema=False)
async def editar_entrada(
    entrada_id: int,
    quantidade: int = Form(...),
    destinatario: str = Form(...),
    motivo: str = Form(""),
    custo_unitario: float = Form(0.0),
    nota_fiscal: UploadFile = File(None),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    entrada = db.get(Entrada, entrada_id)
    if not entrada:
        raise HTTPException(status_code=404, detail="Entrada nao encontrada")
    item = db.get(Item, entrada.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    delta = quantidade - entrada.quantidade
    item.quantidade = max(0, item.quantidade + delta)
    entrada.quantidade = quantidade
    entrada.destinatario = destinatario
    entrada.motivo = motivo
    if custo_unitario:
        entrada.custo_unitario = custo_unitario
        item.valor_unitario = custo_unitario
    nf_path = save_uploaded_file(nota_fiscal, "nf/entradas", allow_pdf=True)
    foto_path = save_uploaded_file(foto, "uploads/entradas")
    if nf_path:
        entrada.nota_fiscal = nf_path
    if foto_path:
        entrada.foto = foto_path
    db.commit()
    record_audit(
        db,
        "entrada_editada",
        item.id,
        delta,
        actor,
        destinatario=entrada.destinatario,
        custo=entrada.custo_unitario,
    )
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)

@app.post("/entradas/{entrada_id}/deletar", include_in_schema=False)
def deletar_entrada(entrada_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    entrada = db.get(Entrada, entrada_id)
    if not entrada:
        raise HTTPException(status_code=404, detail="Entrada nao encontrada")
    item = db.get(Item, entrada.item_id)
    if item:
        item.quantidade = max(0, item.quantidade - entrada.quantidade)
    db.delete(entrada)
    db.commit()
    record_audit(
        db,
        "entrada_excluida",
        entrada.item_id,
        -entrada.quantidade,
        actor,
        destinatario=entrada.destinatario,
        custo=entrada.custo_unitario,
    )
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


# Saidas
@app.get("/saidas", include_in_schema=False)
def saidas_page(request: Request, db: Session = Depends(get_db), user=Depends(require_user)):
    itens = db.query(Item).order_by(Item.descricao).all()
    try:
        page = max(1, int(request.query_params.get("page", "1") or 1))
    except Exception:
        page = 1
    page_size = 8
    item_id_raw = request.query_params.get("item_id")
    quantidade_raw = request.query_params.get("quantidade")
    filial = (request.query_params.get("filial") or "").strip()
    setor = (request.query_params.get("setor") or "").strip()
    motivo = (request.query_params.get("motivo") or "").strip()

    query_saidas = db.query(Saida).join(Item)
    if item_id_raw:
        try:
            item_id_val = int(item_id_raw)
            query_saidas = query_saidas.filter(Saida.item_id == item_id_val)
        except Exception:
            pass
    if quantidade_raw:
        try:
            qty_val = int(quantidade_raw)
            query_saidas = query_saidas.filter(Saida.quantidade == qty_val)
        except Exception:
            pass
    if filial:
        like = f"%{filial}%"
        query_saidas = query_saidas.filter(Saida.destino.ilike(like))
    if setor:
        like = f"%{setor}%"
        query_saidas = query_saidas.filter(Saida.destino.ilike(like))
    if motivo:
        like = f"%{motivo}%"
        query_saidas = query_saidas.filter(Saida.motivo.ilike(like))

    total_saidas = query_saidas.count()
    total_pages = max(1, (total_saidas + page_size - 1) // page_size)
    page = min(page, total_pages)
    offset = (page - 1) * page_size

    saidas = query_saidas.order_by(Saida.created_at.desc()).offset(offset).limit(page_size).all()
    filiais_options = [f.nome for f in db.query(Filial).order_by(Filial.nome).all()]
    setores_options = [s.nome for s in db.query(Setor).order_by(Setor.nome).all()]
    return templates.TemplateResponse(
        "saidas.html",
        {
            "request": request,
            "itens": itens,
            "saidas": saidas,
            "user": user,
            "filiais_options": filiais_options,
            "setores_options": setores_options,
            "page": page,
            "pages": total_pages,
            "page_size": page_size,
            "total_saidas": total_saidas,
            "busca": "",
            "f_item_id": item_id_raw or "",
            "f_quantidade": quantidade_raw or "",
            "f_filial": filial,
            "f_setor": setor,
            "f_motivo": motivo,
        },
    )


@app.get("/api/saidas/serie")
def saidas_serie(days: int = 30, db: Session = Depends(get_db), user=Depends(require_user)):
    days = max(1, min(int(days or 30), 180))
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days - 1)
    rows = (
        db.query(func.date(Saida.created_at), func.sum(Saida.quantidade))
        .filter(Saida.created_at >= start_date)
        .group_by(func.date(Saida.created_at))
        .order_by(func.date(Saida.created_at))
        .all()
    )
    data_map: Dict[datetime.date, int] = {}
    for date_val, total in rows:
        if isinstance(date_val, datetime.date):
            key = date_val
        else:
            try:
                key = datetime.datetime.fromisoformat(str(date_val)).date()
            except Exception:
                continue
        data_map[key] = int(total or 0)

    labels: List[str] = []
    valores: List[int] = []
    for i in range(days):
        day = start_date + datetime.timedelta(days=i)
        labels.append(day.strftime("%d/%m"))
        valores.append(data_map.get(day, 0))

    return {"labels": labels, "quantidades": valores, "total": sum(valores)}


@app.post("/api/saidas")
def registrar_saida_api(payload: Movimento, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(Item, payload.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    if item.quantidade < payload.quantidade:
        raise HTTPException(status_code=400, detail="Quantidade insuficiente")
    item.quantidade -= payload.quantidade
    saida = Saida(
        item_id=item.id,
        quantidade=payload.quantidade,
        destino=payload.destino,
        usuario=actor,
        motivo=payload.motivo,
    )
    db.add(saida)
    db.commit()
    record_audit(db, "saida", item.id, -payload.quantidade, actor, destinatario=payload.destino)
    return {"status": "ok"}


@app.post("/saidas/registrar", include_in_schema=False)
def registrar_saida_form(
    item_id: int = Form(...),
    quantidade: int = Form(...),
    destino: str = Form(""),
    motivo: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    if item.quantidade < quantidade:
        raise HTTPException(status_code=400, detail="Quantidade insuficiente")
    item.quantidade -= quantidade
    saida = Saida(item_id=item.id, quantidade=quantidade, destino=destino, usuario=actor, motivo=motivo)
    db.add(saida)
    db.commit()
    record_audit(db, "saida", item.id, -quantidade, actor, destinatario=destino)
    return RedirectResponse(url="/saidas", status_code=303)


@app.post("/saidas/{saida_id}/deletar", include_in_schema=False)
def deletar_saida(saida_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    saida = db.get(Saida, saida_id)
    if not saida:
        raise HTTPException(status_code=404, detail="Saida nao encontrada")
    item = db.get(Item, saida.item_id)
    if item:
        item.quantidade += saida.quantidade
    db.delete(saida)
    db.commit()
    record_audit(db, "saida_excluida", saida.item_id, saida.quantidade, actor, destinatario=saida.destino)
    return RedirectResponse(url="/saidas", status_code=303)


@app.get("/transferencias", include_in_schema=False)
def transferencias_page(request: Request, db: Session = Depends(get_db), user=Depends(require_user)):
    itens = db.query(Item).order_by(Item.descricao).all()
    transferencias = db.query(Transferencia).order_by(Transferencia.created_at.desc()).limit(20).all()
    return templates.TemplateResponse("transferencias.html", {"request": request, "itens": itens, "transferencias": transferencias, "user": user})


@app.post("/api/transferencias")
def registrar_transferencia_api(payload: TransferCreate, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(Item, payload.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    transferencia = Transferencia(
        item_id=item.id,
        origem=payload.origem or item.localizacao,
        destino=payload.destino,
        quantidade=payload.quantidade,
    )
    item.localizacao = payload.destino
    db.add(transferencia)
    db.commit()
    record_audit(db, "transferencia", item.id, payload.quantidade, actor, destinatario=payload.destino)
    return {"status": "ok"}


@app.post("/transferencias/registrar", include_in_schema=False)
def registrar_transferencia_form(
    item_id: int = Form(...),
    origem: str = Form(""),
    destino: str = Form(...),
    quantidade: int = Form(...),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    transferencia = Transferencia(
        item_id=item.id,
        origem=origem or item.localizacao,
        destino=destino,
        quantidade=quantidade,
    )
    item.localizacao = destino
    db.add(transferencia)
    db.commit()
    record_audit(db, "transferencia", item.id, quantidade, actor, destinatario=destino)
    return RedirectResponse(url="/transferencias", status_code=303)


# Compras
@app.get("/compras", include_in_schema=False)
def compras_page(
    request: Request,
    fornecedor: str = "",
    status: str = "",
    data_pedido: str = "",
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    # mantido redirecionamento original
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


def sync_nf(compra: Compra, nf_path: Optional[str], db: Session):
    if nf_path:
        compra.caminho_nf = nf_path
        db.commit()


def handle_entrega_if_needed(db: Session, compra: Compra, actor: str):
    if compra.status == "entregue":
        item = process_entrega_compra(db, compra, actor)
        if compra.entrada_gerada and item:
            record_audit(
                db,
                "compra_entregue",
                item.id if item else None,
                compra.quantidade,
                actor,
                destinatario=compra.destinatario,
                fornecedor=compra.fornecedor,
                custo=compra.custo_unitario,
            )


@app.post("/compras/criar", include_in_schema=False)
async def compras_criar(
    fornecedor: str = Form(...),
    categoria: str = Form(""),
    item_nome: str = Form(...),
    destino: str = Form(""),
    destinatario: str = Form(...),
    quantidade: int = Form(...),
    data_pedido: str = Form(""),
    custo_unitario: float = Form(...),
    status: str = Form(...),
    nota_fiscal: UploadFile = File(None),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    status_norm = status.lower()
    compra = Compra(
        fornecedor=fornecedor,
        categoria=categoria or None,
        item_nome=item_nome,
        destino=destino,
        destinatario=destinatario,
        usuario=actor,
        quantidade=quantidade,
        data_pedido=data_pedido,
        custo_unitario=custo_unitario,
        status=status_norm,
    )
    db.add(compra)
    db.commit()
    db.refresh(compra)
    record_audit(
        db,
        "compra_criada",
        None,
        compra.quantidade,
        actor,
        destinatario=destinatario,
        fornecedor=fornecedor,
        custo=custo_unitario,
    )
    nf_path = save_uploaded_file(nota_fiscal, f"nf/compras/{compra.id}", allow_pdf=True)
    foto_path = save_uploaded_file(foto, f"uploads/compras/{compra.id}")
    if nf_path:
        sync_nf(compra, nf_path, db)
        record_audit(db, "upload_nf", None, None, actor)
    if foto_path:
        compra.foto = foto_path
        db.commit()
    handle_entrega_if_needed(db, compra, actor)
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


@app.post("/compras/{compra_id}/editar", include_in_schema=False)
async def compras_editar(
    compra_id: int,
    fornecedor: str = Form(...),
    categoria: str = Form(""),
    item_nome: str = Form(...),
    destino: str = Form(""),
    destinatario: str = Form(...),
    quantidade: int = Form(...),
    data_pedido: str = Form(""),
    custo_unitario: float = Form(...),
    status: str = Form(...),
    nota_fiscal: UploadFile = File(None),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    actor = get_actor(user)
    compra = db.get(Compra, compra_id)
    if not compra:
        raise HTTPException(status_code=404, detail="Compra nao encontrada")
    compra.fornecedor = fornecedor
    compra.categoria = categoria or None
    compra.item_nome = item_nome
    compra.destino = destino
    compra.destinatario = destinatario
    compra.quantidade = quantidade
    compra.data_pedido = data_pedido
    compra.custo_unitario = custo_unitario
    compra.status = status.lower()
    compra.usuario = actor
    db.commit()
    if nota_fiscal and nota_fiscal.filename:
        nf_path = save_uploaded_file(nota_fiscal, f"nf/compras/{compra.id}", allow_pdf=True)
        sync_nf(compra, nf_path, db)
        record_audit(db, "upload_nf", None, None, actor)
    if foto and foto.filename:
        foto_path = save_uploaded_file(foto, f"uploads/compras/{compra.id}")
        if foto_path:
            compra.foto = foto_path
            db.commit()
    handle_entrega_if_needed(db, compra, actor)
    record_audit(
        db,
        "compra_atualizada",
        None,
        compra.quantidade,
        actor,
        destinatario=destinatario,
        fornecedor=fornecedor,
        custo=custo_unitario,
    )
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


@app.post("/compras/{compra_id}/deletar", include_in_schema=False)
def compras_deletar(compra_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    compra = db.get(Compra, compra_id)
    if not compra:
        raise HTTPException(status_code=404, detail="Compra nao encontrada")
    db.delete(compra)
    db.commit()
    record_audit(
        db,
        "compra_excluida",
        None,
        None,
        actor,
        destinatario=compra.destinatario,
        fornecedor=compra.fornecedor,
        custo=compra.custo_unitario,
    )
    return RedirectResponse(url="/movimentacoes/entradas", status_code=303)


@app.get("/movimentacoes/entradas", include_in_schema=False)
def movimentacoes_page(
    request: Request,
    tipo: str = "",
    fornecedor: str = "",
    categoria: str = "",
    item: str = "",
    status: str = "",
    destinatario: str = "",
    data: str = "",
    page: int = 1,
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    itens = db.query(Item).order_by(Item.descricao).all()
    compras_query = db.query(Compra)
    entradas_query = db.query(Entrada)
    fornecedores_options = [f.nome for f in db.query(Fornecedor).order_by(Fornecedor.nome).all()]
    categorias_options = [c.nome for c in db.query(Categoria).order_by(Categoria.nome).all()]

    if fornecedor:
        compras_query = compras_query.filter(Compra.fornecedor.ilike(f"%{fornecedor}%"))
    if categoria:
        compras_query = compras_query.filter(Compra.categoria.ilike(f"%{categoria}%"))
    if item:
        compras_query = compras_query.filter(Compra.item_nome.ilike(f"%{item}%"))
        entradas_query = entradas_query.join(Item).filter(Item.descricao.ilike(f"%{item}%"))
    if status:
        compras_query = compras_query.filter(Compra.status == status.lower())
    if destinatario:
        compras_query = compras_query.filter(Compra.destinatario.ilike(f"%{destinatario}%"))
        entradas_query = entradas_query.filter(Entrada.destinatario.ilike(f"%{destinatario}%"))
    if data:
        try:
            data_ref = datetime.datetime.fromisoformat(data).date()
            inicio = datetime.datetime.combine(data_ref, datetime.time.min)
            fim = inicio + datetime.timedelta(days=1)
            entradas_query = entradas_query.filter(Entrada.created_at >= inicio, Entrada.created_at < fim)
            compras_query = compras_query.filter(Compra.data_pedido == data)
        except Exception:
            pass

    entradas = entradas_query.order_by(Entrada.created_at.desc()).all()
    compras = compras_query.order_by(Compra.id.desc()).all()
    compra_map = {c.id: c for c in compras}
    filiais_options = [f.nome for f in db.query(Filial).order_by(Filial.nome).all()]
    setores_options = [s.nome for s in db.query(Setor).order_by(Setor.nome).all()]

    include_manual = tipo in ("", "entrada", "entrada_manual", "manual", "todos")
    include_auto = tipo in ("", "entrada", "entrada_automatica", "automatica", "todos")
    include_compra = tipo in ("", "compra", "todos")

    def normalize_date(raw) -> Optional[datetime.datetime]:
        if isinstance(raw, datetime.datetime):
            return raw
        if isinstance(raw, str) and raw:
            try:
                return datetime.datetime.fromisoformat(raw)
            except Exception:
                return None
        return None

    movimentos = []

    if include_compra:
        for c in compras:
            data_ref = c.data_recebimento or normalize_date(c.data_pedido)
            movimentos.append(
                {
                    "tipo": "Compra",
                    "item_nome": c.item_nome,
                    "categoria": c.categoria,
                    "quantidade": c.quantidade,
                    "destinatario": c.destinatario,
                    "usuario": c.usuario,
                    "fornecedor": c.fornecedor,
                    "destino": c.destino,
                    "custo": c.custo_unitario,
                    "data_mov": data_ref,
                    "status": c.status,
                    "nota_fiscal": c.caminho_nf,
                    "compra_id": c.id,
                    "entrada_id": None,
                    "foto": c.foto,
                }
            )

    for e in entradas:
        is_auto = bool(e.compra_id)
        if is_auto and not include_auto:
            continue
        if (not is_auto) and not include_manual:
            continue
        compra_rel = compra_map.get(e.compra_id) if e.compra_id else None
        if e.compra_id and (fornecedor or status) and not compra_rel:
            continue
        if compra_rel and status and compra_rel.status != status.lower():
            continue
        if status and not compra_rel:
            continue
        movimentos.append(
            {
                "tipo": "Entrada Automatica" if is_auto else "Entrada Manual",
                "item_nome": e.item.descricao if e.item else e.item_id,
                "quantidade": e.quantidade,
                "destinatario": e.destinatario,
                "usuario": e.usuario,
                "fornecedor": compra_rel.fornecedor if compra_rel else None,
                "custo": e.custo_unitario or (compra_rel.custo_unitario if compra_rel else None),
                "data_mov": e.created_at,
                "status": compra_rel.status if compra_rel else "registrada",
                "nota_fiscal": e.nota_fiscal or (compra_rel.caminho_nf if compra_rel else None),
                "compra_id": compra_rel.id if compra_rel else None,
                "entrada_id": e.id,
                "foto": e.foto or (e.item.foto if e.item else None) or (compra_rel.foto if compra_rel else None),
            }
        )

    movimentos = sorted(movimentos, key=lambda m: m["data_mov"] or datetime.datetime.min, reverse=True)

    per_page = 10
    total = len(movimentos)
    total_pages = max((total + per_page - 1) // per_page, 1)
    page = min(max(page, 1), total_pages)
    start = (page - 1) * per_page
    movimentos_paginados = movimentos[start : start + per_page]

    return templates.TemplateResponse(
        "movimentacoes.html",
        {
            "request": request,
            "user": user,
            "itens": itens,
            "movimentos": movimentos_paginados,
            "filtro_tipo": tipo,
            "filtro_fornecedor": fornecedor,
            "filtro_categoria": categoria,
            "filtro_item": item,
            "filtro_status": status,
            "filtro_destinatario": destinatario,
            "filtro_data": data,
            "filiais_options": filiais_options,
            "setores_options": setores_options,
            "fornecedores_options": fornecedores_options,
            "categorias_options": categorias_options,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
    )


# Auditoria
@app.get("/auditoria", include_in_schema=False)
def audit_page(
    request: Request,
    operacao: str = "",
    item: str = "",
    usuario: str = "",
    destinatario: str = "",
    fornecedor: str = "",
    data: str = "",
    page: int = 1,
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    query = db.query(AuditLog)

    if operacao:
        query = query.filter(func.lower(AuditLog.operacao).like(f"%{operacao.lower()}%"))
    if usuario:
        query = query.filter(func.lower(AuditLog.usuario).like(f"%{usuario.lower()}%"))
    if destinatario:
        query = query.filter(func.lower(AuditLog.destinatario).like(f"%{destinatario.lower()}%"))
    if fornecedor:
        query = query.filter(func.lower(AuditLog.fornecedor).like(f"%{fornecedor.lower()}%"))
    if item:
        query = query.join(Item, Item.id == AuditLog.item_id, isouter=True)
        joined_item = True
        item_val = f"%{item.lower()}%"
        query = query.filter(
            func.coalesce(func.lower(Item.descricao), "").
            like(item_val) | func.cast(AuditLog.item_id, String).like(item_val)
        )
    if data:
        try:
            data_ref = datetime.date.fromisoformat(data)
            inicio = datetime.datetime.combine(data_ref, datetime.time.min)
            fim = inicio + datetime.timedelta(days=1)
            query = query.filter(AuditLog.created_at >= inicio, AuditLog.created_at < fim)
        except Exception:
            pass

    page = max(page, 1)
    per_page = 15
    total = query.count()
    logs = (
        query.order_by(AuditLog.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    items_map = {it.id: it.descricao for it in db.query(Item.id, Item.descricao)}
    return templates.TemplateResponse(
        "auditoria.html",
        {
            "request": request,
            "logs": logs,
            "user": user,
            "items_map": items_map,
            "filtro_operacao": operacao,
            "filtro_item": item,
            "filtro_usuario": usuario,
            "filtro_destinatario": destinatario,
            "filtro_fornecedor": fornecedor,
            "filtro_data": data,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page if per_page else 1,
        },
    )


@app.get("/api/audit")
def audit_list_api(db: Session = Depends(get_db), user=Depends(require_admin)):
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).all()
    items_map = {it.id: it.descricao for it in db.query(Item.id, Item.descricao)}
    return [
        {
            "id": log.id,
            "operacao": log.operacao,
            "item_id": log.item_id,
            "item_nome": items_map.get(log.item_id),
            "quantidade": log.quantidade,
            "usuario": log.usuario,
            "destinatario": log.destinatario,
            "fornecedor": log.fornecedor,
            "custo": log.custo,
            "created_at": log.created_at,
        }
        for log in logs
    ]


# Usuários
@app.get("/usuarios", include_in_schema=False)
def users_page(request: Request, db: Session = Depends(get_db), user=Depends(require_admin)):
    users = db.query(Usuario).order_by(Usuario.nome).all()
    return templates.TemplateResponse(
        "usuarios.html",
        {
            "request": request,
            "users": users,
            "user": user,
            "erro": request.query_params.get("erro"),
            "ok": request.query_params.get("ok"),
        },
    )


@app.post("/usuarios/criar", include_in_schema=False)
async def create_user(
    nome: str = Form(...),
    email: str = Form(...),
    funcao: str = Form("comum"),
    senha: str = Form(...),
    confirmar_senha: str = Form(...),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    if senha != confirmar_senha:
        return RedirectResponse(url="/usuarios?erro=senhas", status_code=303)
    if db.query(Usuario).filter(Usuario.email == email).first():
        return RedirectResponse(url="/usuarios?erro=email", status_code=303)
    foto_path = save_uploaded_file(foto, "uploads/usuarios")
    novo = Usuario(nome=nome, email=email, funcao=funcao, senha_hash=hash_password(senha), foto=foto_path)
    db.add(novo)
    db.commit()
    return RedirectResponse(url="/usuarios?ok=criado", status_code=303)


@app.post("/usuarios/{user_id}/editar", include_in_schema=False)
async def edit_user(
    user_id: int,
    nome: str = Form(...),
    funcao: str = Form(...),
    senha: str = Form(""),
    confirmar_senha: str = Form(""),
    foto: UploadFile = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    target = db.get(Usuario, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="Usuario nao encontrado")
    target.nome = nome
    target.funcao = funcao
    if senha or confirmar_senha:
        if senha != confirmar_senha:
            return RedirectResponse(url="/usuarios?erro=senhas", status_code=303)
        target.senha_hash = hash_password(senha)
    if foto and foto.filename:
        foto_path = save_uploaded_file(foto, "uploads/usuarios")
        target.foto = foto_path
    db.commit()
    return RedirectResponse(url="/usuarios?ok=editado", status_code=303)

@app.post("/usuarios/{user_id}/deletar", include_in_schema=False)
def delete_user_api(user_id: int, db: Session = Depends(get_db), user=Depends(require_admin)):
    target = db.get(Usuario, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    if target.funcao == "admin":
        admins = db.query(Usuario).filter(Usuario.funcao == "admin").count()
        if admins <= 1:
            return RedirectResponse(url="/usuarios?erro=ultimo_admin", status_code=303)
    db.delete(target)
    db.commit()
    return RedirectResponse(url="/usuarios?ok=excluido", status_code=303)


def format_datetime(value: Optional[datetime.datetime]):
    if not value:
        return ""
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        value = value.astimezone(tz)
    except Exception:
        pass
    return value.strftime("%d/%m/%Y %H:%M")


def format_currency(value: Optional[float]):
    if value is None:
        return "-"
    try:
        return ("R$ {:,.2f}".format(float(value))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"R$ {value}"


templates.env.filters["datetime"] = format_datetime
templates.env.filters["real"] = format_currency


@app.get("/login", include_in_schema=False)
def login_page(request: Request, session: Optional[str] = Cookie(None)):
    if session and session in SESSION_STORE:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", include_in_schema=False)
def login_action(request: Request, email: str = Form(...), senha: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.email == email).first()
    if not user or not verify_password(senha, user.senha_hash):
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Credenciais invalidas"}, status_code=400
        )
    token = create_session(user)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie("session", token, httponly=True, samesite="lax")
    return response


@app.get("/logout", include_in_schema=False)
def logout(session: Optional[str] = Cookie(None)):
    clear_session(session)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session")
    return response


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}
