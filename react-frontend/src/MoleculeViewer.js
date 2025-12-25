import React, { useEffect, useRef } from "react";

export default function MoleculeViewer({ smiles }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!smiles) return;

    async function drawMolecule() {
      // Wait for RDKit to load (from script in index.html)
      if (!window.initRDKitModule) {
        console.error("RDKit script not loaded");
        return;
      }

      const RDKit = await window.initRDKitModule();

      try {
        const mol = RDKit.get_mol(smiles);

        // RDKit requires the canvas element ID
        const canvas = canvasRef.current;

        mol.draw_to_canvas(canvas, 300, 300);
      } catch (e) {
        console.error("RDKit failed:", e);
      }
    }

    drawMolecule();
  }, [smiles]);

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <canvas
        ref={canvasRef}
        width={300}
        height={300}
        style={{
          border: "1px solid #ccc",
          borderRadius: "8px",
          margin: "auto",
        }}
      />
    </div>
  );
}
