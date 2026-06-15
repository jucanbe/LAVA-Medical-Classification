from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Frontend"])

TEMPLATES_DIR = Path(__file__).parent.parent / "frontend" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_page": "home"
    })


@router.get("/app/docs", response_class=HTMLResponse, include_in_schema=False)
async def documentation(request: Request):
    """User documentation page."""
    return templates.TemplateResponse("docs.html", {
        "request": request,
        "active_page": "docs"
    })


@router.get("/app/llm/classify", response_class=HTMLResponse, include_in_schema=False)
async def llm_classify(request: Request):
    """LLM classification page."""
    return templates.TemplateResponse("llm/classify.html", {
        "request": request,
        "active_page": "llm-classify"
    })


@router.get("/app/llm/relations", response_class=HTMLResponse, include_in_schema=False)
async def llm_relations(request: Request):
    """LLM relation classification page."""
    return templates.TemplateResponse("llm/relations.html", {
        "request": request,
        "active_page": "llm-relations"
    })


@router.get("/app/llm/classify-all", response_class=HTMLResponse, include_in_schema=False)
async def llm_classify_all(request: Request):
    """LLM Classify All (LangExtract) page."""
    return templates.TemplateResponse("llm/classify_all.html", {
        "request": request,
        "active_page": "llm-classify-all"
    })


@router.get("/app/llm/export", response_class=HTMLResponse, include_in_schema=False)
async def llm_export(request: Request):
    """LLM IOB export page."""
    return templates.TemplateResponse("llm/export.html", {
        "request": request,
        "active_page": "llm-export"
    })


@router.get("/app/bert/classify", response_class=HTMLResponse, include_in_schema=False)
async def bert_classify(request: Request):
    """BERT classification page."""
    return templates.TemplateResponse("bert/classify.html", {
        "request": request,
        "active_page": "bert-classify"
    })


@router.get("/app/bert/relations", response_class=HTMLResponse, include_in_schema=False)
async def bert_relations(request: Request):
    """BERT relation classification page."""
    return templates.TemplateResponse("bert/relations.html", {
        "request": request,
        "active_page": "bert-relations"
    })


@router.get("/app/bert/train", response_class=HTMLResponse, include_in_schema=False)
async def bert_train(request: Request):
    """BERT training page."""
    return templates.TemplateResponse("bert/train.html", {
        "request": request,
        "active_page": "bert-train"
    })


@router.get("/app/bert/models", response_class=HTMLResponse, include_in_schema=False)
async def bert_models(request: Request):
    """BERT models management page."""
    return templates.TemplateResponse("bert/models.html", {
        "request": request,
        "active_page": "bert-models"
    })


@router.get("/app/entities/pending", response_class=HTMLResponse, include_in_schema=False)
async def entities_pending(request: Request):
    """Pending entities page."""
    return templates.TemplateResponse("entities/pending.html", {
        "request": request,
        "active_page": "entities-pending"
    })


@router.get("/app/entities/review", response_class=HTMLResponse, include_in_schema=False)
async def entities_review(request: Request):
    """Entity review scorecard page."""
    return templates.TemplateResponse("entities/review.html", {
        "request": request,
        "active_page": "entities-review"
    })


@router.get("/app/relations/pending", response_class=HTMLResponse, include_in_schema=False)
async def relations_pending(request: Request):
    """Pending relations page."""
    return templates.TemplateResponse("relations/pending.html", {
        "request": request,
        "active_page": "relations-pending"
    })


@router.get("/app/relations/review", response_class=HTMLResponse, include_in_schema=False)
async def relations_review(request: Request):
    """Relation review page."""
    return templates.TemplateResponse("relations/review.html", {
        "request": request,
        "active_page": "relations-review"
    })


@router.get("/app/kg/overview", response_class=HTMLResponse, include_in_schema=False)
async def kg_overview(request: Request):
    """Knowledge Graph overview page."""
    return templates.TemplateResponse("kg/overview.html", {
        "request": request,
        "active_page": "kg-overview"
    })


@router.get("/app/kg/sparql", response_class=HTMLResponse, include_in_schema=False)
async def kg_sparql(request: Request):
    """SPARQL query page."""
    return templates.TemplateResponse("kg/sparql.html", {
        "request": request,
        "active_page": "kg-sparql"
    })


@router.get("/app/config/llm", response_class=HTMLResponse, include_in_schema=False)
async def config_llm(request: Request):
    """LLM configuration page."""
    return templates.TemplateResponse("config/llm.html", {
        "request": request,
        "active_page": "config-llm"
    })


@router.get("/app/config/kg", response_class=HTMLResponse, include_in_schema=False)
async def config_kg(request: Request):
    """Knowledge Graph configuration page — redirects to KG Overview."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/kg/overview", status_code=301)


@router.get("/app/config/triple-store", response_class=HTMLResponse, include_in_schema=False)
async def config_triple_store(request: Request):
    """Triple Store configuration page."""
    return templates.TemplateResponse("config/triple_store.html", {
        "request": request,
        "active_page": "config-triplestore"
    })


@router.get("/app/config/entity", response_class=HTMLResponse, include_in_schema=False)
async def config_entity(request: Request):
    """Entity configuration page for scorecard settings."""
    return templates.TemplateResponse("config/entity.html", {
        "request": request,
        "active_page": "config-entity"
    })


@router.get("/app/config/relation", response_class=HTMLResponse, include_in_schema=False)
async def config_relation(request: Request):
    """Relation configuration page for scorecard settings."""
    return templates.TemplateResponse("config/relation.html", {
        "request": request,
        "active_page": "config-relation"
    })


@router.get("/app/config/status", response_class=HTMLResponse, include_in_schema=False)
async def config_status(request: Request):
    """System status page."""
    return templates.TemplateResponse("config/status.html", {
        "request": request,
        "active_page": "config-status"
    })
