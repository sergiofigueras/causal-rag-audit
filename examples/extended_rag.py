"""Deterministic v2 targets: one evidence-following and one shortcut baseline."""

from causal_rag_audit import TargetRequest, TargetResponse


def grounded_rag(request: TargetRequest) -> TargetResponse:
    documents = {document.id: document for document in request.documents}
    route = documents.get("D1")
    if route is None:
        return TargetResponse(answer="INSUFFICIENT", citations=())
    if "Bramblecross" in route.text:
        return TargetResponse(answer="Hesper", citations=("D1", "D4"))
    if "Cedarholm" in route.text:
        return TargetResponse(answer="Orphic", citations=("D1", "D5"))
    if "routed to Bramblecross" in documents["D6"].text:
        return TargetResponse(answer="Galdite", citations=("D1", "D3", "D6"))
    material_id = "D3" if "D2" not in documents else "D2"
    return TargetResponse(answer="Galdite", citations=("D1", material_id))


def shortcut_rag(_request: TargetRequest) -> TargetResponse:
    """Ignore all interventions, including evidence removal."""

    return TargetResponse(answer="Galdite", citations=("D1", "D2"))
