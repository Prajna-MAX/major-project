import React from "react";

export default function ExplanationCard({ results }) {
  if (!results) return null;

  const mediumRisk = [];
  const toxic = [];

  Object.entries(results.predictions).forEach(([endpoint, obj]) => {
    if (obj.risk_level === "Medium Risk") mediumRisk.push(endpoint);
    if (obj.risk_level === "Toxic") toxic.push(endpoint);
  });

  // Text generation
  const hasToxic = toxic.length > 0;
  const hasMedium = mediumRisk.length > 0;

  return (
    <div
      style={{
        marginTop: "30px",
        padding: "20px",
        background: "#fff8e6",
        borderRadius: "12px",
        border: "1px solid #ffd27f",
      }}
    >
      <h2 style={{ marginBottom: "10px" }}>📘 Toxicity Explanation</h2>

      {!hasToxic && !hasMedium && (
        <p style={{ fontSize: "1rem" }}>
          ✅ This compound is **predicted to be non-toxic** across all tested
          biological pathways. No significant risks were detected.
        </p>
      )}

      {hasMedium && (
        <>
          <p style={{ fontSize: "1rem", marginBottom: "10px" }}>
            ⚠ The model detected **moderate risk** in the following pathways:
          </p>
          <ul style={{ paddingLeft: "20px" }}>
            {mediumRisk.map((e) => (
              <li key={e} style={{ marginBottom: "6px" }}>
                <b>{e}</b> — this suggests potential functional stress
                (oxidative, mitochondrial, or cellular response).
              </li>
            ))}
          </ul>
        </>
      )}

      {hasToxic && (
        <>
          <p style={{ fontSize: "1rem", marginTop: "10px", marginBottom: "10px" }}>
            ❗ This compound shows **high toxicity risk** in these pathways:
          </p>
          <ul style={{ paddingLeft: "20px" }}>
            {toxic.map((e) => (
              <li key={e} style={{ color: "#d9534f", marginBottom: "6px" }}>
                <b>{e}</b> — indicates a strong toxic effect that may impact
                biological systems.
              </li>
            ))}
          </ul>
        </>
      )}

      <hr style={{ margin: "20px 0", opacity: 0.3 }} />

      <p style={{ fontSize: "0.95rem", lineHeight: "1.5" }}>
        This explanation card summarizes the predicted toxicity based on 12
        Tox21 biological endpoints, including hormone disruption, DNA damage,
        oxidative stress, and mitochondrial effects.  
        It helps interpret whether a compound is generally safe, moderately
        risky, or potentially harmful.
      </p>
    </div>
  );
}
