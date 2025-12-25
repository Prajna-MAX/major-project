import React from "react";

export default function ToxicFragmentHighlight({ smiles }) {
  if (!smiles) return null;

  // Simple example toxic fragments
  const fragments = [
    { fragment: "Cl", reason: "Halogen" },
    { fragment: "Br", reason: "Halogen" },
    { fragment: "NO2", reason: "Nitro group" },
    { fragment: "N=N", reason: "Azo group" },
    { fragment: "O=O", reason: "Peroxide" },
  ];

  const detected = fragments.filter(f => smiles.includes(f.fragment));

  return (
    <div style={{ marginTop: "25px" }}>
      <h3>Detected Toxic Fragments</h3>

      {detected.length === 0 ? (
        <p style={{ color: "#555" }}>No known toxic fragments found.</p>
      ) : (
        <ul>
          {detected.map((f, i) => (
            <li key={i}>
              <b>{f.fragment}</b> — {f.reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
