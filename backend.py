# backend.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import GNN_ToxModel, mol_to_graph

app = Flask(__name__)
CORS(app)

TARGET_COLS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# ---- Custom thresholds tuned for GNN ---- #
THRESHOLDS = {
    "NR-AR": 0.05,
    "NR-AR-LBD": 0.02,
    "NR-AhR": 0.10,
    "NR-Aromatase": 0.10,
    "NR-ER": 0.15,
    "NR-ER-LBD": 0.03,
    "NR-PPAR-gamma": 0.20,
    "SR-ARE": 0.50,
    "SR-ATAD5": 0.18,
    "SR-HSE": 0.30,
    "SR-MMP": 0.27,
    "SR-p53": 0.30
}

def classify(prob, thr):
    if prob >= thr:
        return 1     # toxic
    elif prob >= thr * 0.6:
        return -1    # medium risk
    else:
        return 0     # safe


# Load trained model
device = torch.device("cpu")
model = GNN_ToxModel()
model.load_state_dict(torch.load("gnn_best.pt", map_location=device))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    smiles = data.get("smiles")

    graph = mol_to_graph(smiles)
    if graph is None:
        return jsonify({"error": "Invalid SMILES"}), 400

    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

    with torch.no_grad():
        logits = model(graph.unsqueeze(0)) if hasattr(graph, "unsqueeze") else model(graph)

    predictions = {}
    for i, target in enumerate(TARGET_COLS):
        prob = float(torch.sigmoid(logits[0][i]).item())
        thr = THRESHOLDS[target]
        cls = classify(prob, thr)

        predictions[target] = {
            "probability": prob,
            "prediction": cls,
            "risk_level": (
                "Toxic" if cls == 1 else
                "Medium Risk" if cls == -1 else
                "Non-Toxic"
            )
        }

    return jsonify({
        "smiles": smiles,
        "predictions": predictions
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
