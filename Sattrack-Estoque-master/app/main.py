import datetime
import uuid
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional

import bcrypt
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine, func, text
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# Database setup
DATABASE_URL = "sqlite:///./sattrack.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

SESSION_STORE: Dict[str, Dict[str, str]] = {}
DEFAULT_ADMIN_EMAIL = "admin@sattrack.local"
DEFAULT_ADMIN_PASSWORD = "admin"
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


class Compra(Base):
    __tablename__ = "compras"

    id = Column(Integer, primary_key=True, index=True)
    fornecedor = Column(String, nullable=False)
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
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_compras_status ON compras(status)"))


def ensure_item_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(items)"))}
        if "foto" not in cols:
            conn.execute(text("ALTER TABLE items ADD COLUMN foto TEXT"))


def ensure_audit_schema():
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(audit_log)"))}
        if "destinatario" not in cols:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN destinatario TEXT"))
        if "fornecedor" not in cols:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN fornecedor TEXT"))
        if "custo" not in cols:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN custo REAL"))


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
    return get_actor(user) or "sistema"


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


def find_or_create_item_for_compra(db: Session, nome: str, custo_unitario: float, user_label: str) -> Item:
    item = db.query(Item).filter(func.lower(Item.descricao) == nome.lower()).first()
    if item:
        if custo_unitario:
            item.valor_unitario = custo_unitario
            db.commit()
        return item
    base_codigo = "".join(ch for ch in nome.upper().replace(" ", "_") if ch.isalnum() or ch == "_") or "ITEM"
    codigo = base_codigo
    suffix = 1
    while db.query(Item).filter(Item.codigo_interno == codigo).first():
        suffix += 1
        codigo = f"{base_codigo}_{suffix}"
    item = Item(
        codigo_interno=codigo,
        descricao=nome,
        categoria="Sem Categoria",
        quantidade=0,
        valor_unitario=custo_unitario or 0.0,
        localizacao="Almoxarifado",
    )
    db.add(item)
    db.commit()
    record_audit(db, "cadastro_item_por_compra", item.id, 0, user_label, custo=custo_unitario)
    return item


def process_entrega_compra(db: Session, compra: Compra, user_label: str):
    if compra.entrada_gerada:
        return
    item = find_or_create_item_for_compra(db, compra.item_nome, compra.custo_unitario, user_label)
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
    hoje = datetime.datetime.utcnow().date()
    labels: List[str] = []
    entradas: List[int] = []
    saidas: List[int] = []
    for i in range(6, -1, -1):
        dia = hoje - datetime.timedelta(days=i)
        inicio = datetime.datetime.combine(dia, datetime.time.min)
        fim = inicio + datetime.timedelta(days=1)
        total_entradas = (
            db.query(func.coalesce(func.sum(Entrada.quantidade), 0))
            .filter(Entrada.created_at >= inicio, Entrada.created_at < fim)
            .scalar()
            or 0
        )
        total_saidas = (
            db.query(func.coalesce(func.sum(Saida.quantidade), 0))
            .filter(Saida.created_at >= inicio, Saida.created_at < fim)
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

# Inventory pages
@app.get("/inventario", include_in_schema=False)
def inventory_page(
    request: Request,
    codigo: str = "",
    descricao: str = "",
    categoria: str = "",
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
    itens = query.order_by(Item.descricao).all()
    categorias = [row[0] for row in db.query(Item.categoria).filter(Item.categoria.isnot(None)).distinct().order_by(Item.categoria)]
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
    foto_path = save_uploaded_file(foto, "uploads/itens")
    if foto_path:
        item.foto = foto_path
    db.commit()
    record_audit(db, "edicao", item.id, delta, actor)
    return RedirectResponse(url="/inventario", status_code=303)


@app.post("/items/{item_id}/deletar", include_in_schema=False)
def delete_item_form(item_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
    actor = get_actor(user)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado")
    db.delete(item)
    db.commit()
    record_audit(db, "exclusao", item_id, None, actor)
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
    saidas = db.query(Saida).order_by(Saida.created_at.desc()).limit(20).all()
    return templates.TemplateResponse("saidas.html", {"request": request, "itens": itens, "saidas": saidas, "user": user})


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
    item: str = "",
    status: str = "",
    destinatario: str = "",
    data: str = "",
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    itens = db.query(Item).order_by(Item.descricao).all()
    compras_query = db.query(Compra)
    entradas_query = db.query(Entrada)

    if fornecedor:
        compras_query = compras_query.filter(Compra.fornecedor.ilike(f"%{fornecedor}%"))
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

    entradas = entradas_query.order_by(Entrada.created_at.desc()).limit(200).all()
    compras = compras_query.order_by(Compra.id.desc()).all()
    compra_map = {c.id: c for c in compras}

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

    movimentos = sorted(movimentos, key=lambda m: m["data_mov"] or datetime.datetime.min, reverse=True)[:200]

    return templates.TemplateResponse(
        "movimentacoes.html",
        {
            "request": request,
            "user": user,
            "itens": itens,
            "movimentos": movimentos,
            "filtro_tipo": tipo,
            "filtro_fornecedor": fornecedor,
            "filtro_item": item,
            "filtro_status": status,
            "filtro_destinatario": destinatario,
            "filtro_data": data,
        },
    )


# Auditoria
@app.get("/auditoria", include_in_schema=False)
def audit_page(request: Request, db: Session = Depends(get_db), user=Depends(require_admin)):
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(50).all()
    items_map = {it.id: it.descricao for it in db.query(Item.id, Item.descricao)}
    return templates.TemplateResponse(
        "auditoria.html",
        {"request": request, "logs": logs, "user": user, "items_map": items_map},
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


templates.env.filters["datetime"] = format_datetime


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
