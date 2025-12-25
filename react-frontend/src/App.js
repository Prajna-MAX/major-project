import React, { useState } from "react";
import MoleculeViewer from "./MoleculeViewer";
import ExplanationCard from "./ExplainationCard";
import "./App.css";   // <-- new CSS file

export default function App() {
  const [compoundName, setCompoundName] = useState("");
  const [smiles, setSmiles] = useState("");
  const [resolvedSmiles, setResolvedSmiles] = useState("");
  const [loadingSmiles, setLoadingSmiles] = useState(false);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const endpointMeaning = {
    "NR-AR": "Androgen",
    "NR-AR-LBD": "AR-Binding",
    "NR-AhR": "Aryl-Hydrocarbons",
    "NR-Aromatase": "Hormones",
    "NR-ER": "Estrogen",
    "NR-ER-LBD": "ER-Binding",
    "NR-PPAR-gamma": "Metabolism",
    "SR-ARE": "Oxidative",
    "SR-ATAD5": "DNA-Repair",
    "SR-HSE": "Heat-Shock",
    "SR-MMP": "Mitochondria",
    "SR-p53": "DNA-Damage",
  };




  async function handlePredict() {
    if (!smiles.trim()) return alert("Please enter a SMILES string!");
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Backend not reachable");
    }

    setLoading(false);
  }

  return (
    <div className="app-container">
      <div className="main-card">
        <h1 className="title"> Toxicity Predictor</h1>


       

        {/* SMILES Input */}
        <label className="label">SMILES:</label>
        <input
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="Example: CCO"
          className="input"
        />

        {smiles && <MoleculeViewer smiles={smiles} />}

        <button className="btn-predict" onClick={handlePredict}>
          {loading ? "Predicting…" : "Predict Toxicity"}
        </button>

        {result && !result.error && (
          <div className="results-section">
            <h2>Results</h2>

            <table className="tox-table">
              <thead>
                <tr>
                 
                  <th>Receptors</th>
                  <th>Risk</th>
                </tr>
              </thead>

              <tbody>
                {Object.entries(result.predictions).map(([endpoint, obj]) => (
                  <tr key={endpoint}>
                    <td>
                      <i>{endpointMeaning[endpoint]}</i>
                    </td>
                    <td>
                      <span className={`risk-badge ${obj.risk_level.replace(" ", "-")}`}>
                        {obj.risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <ExplanationCard results={result} />
          </div>
        )}
      </div>

      {/* Properties Card */}
      {result?.properties && (
        <div className="properties-card">
          <h2> Physicochemical Properties</h2>

          <div className="grid">
            {Object.entries(result.properties).map(([key, value]) => (
              <div className="prop-box" key={key}>
                <div className="prop-key">{key}</div>
                <div className="prop-value">{value.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
