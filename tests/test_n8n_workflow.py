import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "n8n" / "esik-ai-workflow.json"


def test_n8n_workflow_is_valid_json():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as file:
        workflow = json.load(file)

    assert workflow["id"] == "ESIK_OPENAI_V1"
    assert workflow["name"] == "Eşik AI - Finansal Risk Asistanı"
    assert workflow["active"] is False


def test_n8n_workflow_contains_required_nodes():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as file:
        workflow = json.load(file)

    node_types = {node["type"] for node in workflow["nodes"]}
    assert "n8n-nodes-base.webhook" in node_types
    assert "@n8n/n8n-nodes-langchain.chainLlm" in node_types
    assert "@n8n/n8n-nodes-langchain.lmChatOpenAi" in node_types
    assert "@n8n/n8n-nodes-langchain.lmChatGoogleGemini" not in node_types
    assert "n8n-nodes-base.respondToWebhook" in node_types


def test_n8n_workflow_contains_no_exported_credentials():
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert '"credentials"' not in workflow_text
    assert "OPENAI_API_KEY" not in workflow_text


def test_openai_model_is_configured_without_a_secret():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    openai_nodes = [
        node
        for node in workflow["nodes"]
        if node["type"] == "@n8n/n8n-nodes-langchain.lmChatOpenAi"
    ]
    assert len(openai_nodes) == 1
    assert openai_nodes[0]["name"] == "OpenAI Chat Model"
    configured_model = openai_nodes[0]["parameters"]["model"]
    # n8n exports resource-locator fields as objects in current versions.
    if isinstance(configured_model, dict):
        configured_model = configured_model["value"]
    assert configured_model == "gpt-5.6-luna"


def test_workflow_field_names_are_unique_and_connections_resolve():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    names = {node["name"] for node in workflow["nodes"]}
    for node in workflow["nodes"]:
        fields = node["parameters"].get("assignments", {}).get("assignments", [])
        assert len(fields) == len({field["name"] for field in fields})
    for source, connections in workflow["connections"].items():
        assert source in names
        for branches in connections.values():
            for branch in branches:
                for target in branch:
                    assert target["node"] in names
